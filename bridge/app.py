"""
bridge/app.py — The Windows MT5 bridge service.

Runs INSIDE the Windows 11 VM alongside the MetaTrader 5 terminal. It is the
only component permitted to import MetaTrader5.

The Linux core talks to this over HTTP. Strategy, risk and portfolio logic are
deliberately NOT duplicated here — the bridge translates broker calls and
nothing else.

Safety model
------------
* **Read-only by default.** Order submission, modification and closing require
  ``MT5_BRIDGE_ALLOW_WRITES=1`` *and* a valid token. Any state-changing endpoint
  returns 403 otherwise.
* **Writes require a demo/live check.** ``MT5_BRIDGE_EXPECTED_LOGIN`` must be
  set and must match the connected account. A caller cannot assert its own
  account identity.
* **Idempotency.** Repeats of an idempotency key are refused within the cache
  window so a retried HTTP request cannot create a second order.
* **No secrets in responses.** Tokens and MT5 passwords never leave the host.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse

from bridge.auth import BridgeAuth, token_from_env
from bridge.schemas import (
    AccountInfoResponse,
    ActionResponse,
    ClosePositionRequest,
    DataSource,
    DealResponse,
    DealsResponse,
    ErrorResponse,
    HealthResponse,
    ModifyPositionRequest,
    OrderRequestModel,
    OrderResponse,
    OrderValidationResponse,
    PositionResponse,
    PositionsResponse,
    PriceResponse,
    RatesResponse,
    ReadyResponse,
    SymbolInfoResponse,
    SymbolsResponse,
    TerminalState,
    TerminalStatus,
)
from core.execution.states import state_from_retcode
from core.mt5.types import OrderRequest, OrderType, TradeAction

logger = logging.getLogger(__name__)

BRIDGE_VERSION = "1.0.0"
API_PREFIX = "/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Adapter selection
# ─────────────────────────────────────────────────────────────────────────────

def build_adapter():
    """
    Construct the broker adapter appropriate for this host.

    Returns ``RealMT5Adapter`` when MetaTrader5 is importable (the intended
    Windows deployment), otherwise ``MockMT5Adapter`` so the service can be
    started and smoke-tested off-Windows.

    The fallback is announced loudly in the readiness response — it must never
    be mistaken for a live broker connection.
    """
    from core.mt5.adapter import MockMT5Adapter

    try:
        import MetaTrader5  # noqa: F401
    except ImportError:
        logger.error(
            "MetaTrader5 is not available on this host. Starting in MOCK mode — "
            "all data is synthetic and no broker is connected."
        )
        adapter = MockMT5Adapter()
        adapter.connect()
        return adapter

    from core.settings import get_settings
    from core.mt5.adapter import RealMT5Adapter

    settings = get_settings()
    adapter = RealMT5Adapter(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server,
        path=settings.mt5_path,
    )
    adapter.connect()
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency (in-process; see bridge docs for the limitation)
# ─────────────────────────────────────────────────────────────────────────────

class _IdempotencyCache:
    """Remembers recently used keys so a retried request cannot duplicate."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._seen: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            expired = [k for k, t in self._seen.items() if now - t > self._ttl]
            for k in expired:
                self._seen.pop(k, None)
            if key in self._seen:
                return True
            self._seen[key] = now
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(adapter=None, allow_writes: Optional[bool] = None) -> FastAPI:
    """
    Build the bridge application.

    Injectable ``adapter`` and ``allow_writes`` exist so tests can exercise
    every path on Linux without a terminal and without enabling real writes.
    """
    app = FastAPI(
        title="MT5 Bridge",
        version=BRIDGE_VERSION,
        description=(
            "Windows-side MetaTrader 5 bridge. Read-only by default. "
            "Exposes the MT5Adapter contract over HTTP."
        ),
    )

    app.state.adapter = adapter if adapter is not None else build_adapter()
    app.state.auth = BridgeAuth(token_from_env())
    app.state.idempotency = _IdempotencyCache()
    app.state.allow_writes = (
        os.getenv("MT5_BRIDGE_ALLOW_WRITES", "0") == "1"
        if allow_writes is None
        else allow_writes
    )
    app.state.expected_login = os.getenv("MT5_BRIDGE_EXPECTED_LOGIN") or None
    app.state.mock_mode = type(app.state.adapter).__name__ == "MockMT5Adapter"

    auth = app.state.auth

    # ── Guards ──────────────────────────────────────────────────────────────

    def require_writes() -> None:
        """Deny state-changing calls unless explicitly enabled."""
        if not app.state.allow_writes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="order submission is disabled on this bridge "
                       "(set MT5_BRIDGE_ALLOW_WRITES=1 to enable)",
            )

    def require_expected_account() -> None:
        """
        Verify the connected account matches configuration.

        The account is read from terminal metadata, never from the request
        body — a caller must not be able to assert who it is.
        """
        expected = app.state.expected_login
        if expected is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MT5_BRIDGE_EXPECTED_LOGIN is not configured on this host",
            )
        info = app.state.adapter.get_account_info()
        if info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="terminal is not connected",
            )
        if int(info.login) != int(expected):
            logger.error(
                "Account mismatch: expected %s, connected to %s",
                expected, info.login,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"connected account {info.login} does not match "
                       f"configured account {expected}",
            )

    # ── Error shape ─────────────────────────────────────────────────────────

    @app.exception_handler(HTTPException)
    async def http_error(request, exc: HTTPException):
        # Preserve the caller's reason. Discarding it would leave operators
        # with a status code and no explanation, which is exactly the failure
        # mode this error shape exists to prevent.
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=detail,
                detail=detail,
                code=str(exc.status_code),
            ).model_dump(),
        )

    # ── Health / readiness (unauthenticated: no sensitive data) ─────────────

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", service="mt5-bridge", version=BRIDGE_VERSION)

    @app.get(f"{API_PREFIX}/ready", response_model=ReadyResponse)
    def ready():
        """Liveness of the *broker* side. Never authenticated — leaks nothing."""
        if app.state.mock_mode:
            return ReadyResponse(
                ready=False,
                terminal_state=TerminalState.NOT_CONNECTED,
                detail="running in MOCK mode (no MetaTrader5 on this host) — "
                       "no broker connection",
            )
        if not auth.configured:
            return ReadyResponse(
                ready=False,
                terminal_state=TerminalState.NOT_CONNECTED,
                detail="bridge token is not configured",
            )
        info = app.state.adapter.get_account_info()
        if info is None:
            return ReadyResponse(
                ready=False,
                terminal_state=TerminalState.NOT_CONNECTED,
                detail="terminal not connected",
            )
        if app.state.expected_login and int(info.login) != int(app.state.expected_login):
            return ReadyResponse(
                ready=False,
                terminal_state=TerminalState.ACCOUNT_MISMATCH,
                detail=f"connected account {info.login} does not match configuration",
                login=info.login,
                server=info.server,
            )
        return ReadyResponse(
            ready=True,
            terminal_state=TerminalState.LOGGED_IN,
            login=info.login,
            server=info.server,
        )

    @app.get(f"{API_PREFIX}/terminal/status", response_model=TerminalStatus)
    def terminal_status(_: None = Depends(auth.require_token)):
        if app.state.mock_mode:
            return TerminalStatus(
                terminal_state=TerminalState.NOT_CONNECTED,
                last_error="running in MOCK mode",
                source=DataSource.UNAVAILABLE,
            )
        info = app.state.adapter.get_account_info()
        if info is None:
            return TerminalStatus(
                terminal_state=TerminalState.NOT_CONNECTED,
                last_error="no account info",
                source=DataSource.UNAVAILABLE,
            )
        return TerminalStatus(
            terminal_state=TerminalState.LOGGED_IN,
            login=info.login,
            server=info.server,
            account_name=info.name or None,
            source=DataSource.BROKER,
        )

    # ── Account ─────────────────────────────────────────────────────────────

    @app.get(f"{API_PREFIX}/account", response_model=AccountInfoResponse)
    def account(_: None = Depends(auth.require_token)):
        info = app.state.adapter.get_account_info()
        if info is None:
            raise HTTPException(status_code=503, detail="terminal is not connected")
        return AccountInfoResponse(
            login=info.login,
            server=info.server,
            currency=info.currency,
            leverage=info.leverage,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            margin_free=info.margin_free,
            profit=info.profit,
            source=DataSource.BROKER,
        )

    # ── Instruments ─────────────────────────────────────────────────────────

    @app.get(f"{API_PREFIX}/symbols", response_model=SymbolsResponse)
    def symbols(_: None = Depends(auth.require_token)):
        """Dynamically discovered symbols. No hardcoded instrument list."""
        discovery = getattr(app.state.adapter, "get_symbols", None)
        found = list(discovery()) if callable(discovery) else []
        return SymbolsResponse(symbols=found, count=len(found), source=DataSource.BROKER)

    @app.get(f"{API_PREFIX}/symbol/{{symbol}}", response_model=SymbolInfoResponse)
    def symbol_info(symbol: str, _: None = Depends(auth.require_token)):
        info = app.state.adapter.get_symbol_info(symbol)
        if info is None:
            raise HTTPException(status_code=404, detail=f"unknown symbol: {symbol}")
        # Values are passed through verbatim; absence stays None rather than
        # being filled with a plausible default.
        return SymbolInfoResponse(
            name=info.name,
            description=info.description,
            digits=info.digits,
            point=info.point,
            trade_tick_size=info.trade_tick_size,
            trade_tick_value=info.trade_tick_value,
            trade_stops_level=info.trade_stops_level,
            trade_freeze_level=info.trade_freeze_level,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            contract_size=info.contract_size,
            spread=info.spread,
            trade_mode=info.trade_mode,
            source=DataSource.BROKER,
        )

    # ── Market data ─────────────────────────────────────────────────────────

    @app.get(f"{API_PREFIX}/price/{{symbol}}", response_model=PriceResponse)
    def price(symbol: str, _: None = Depends(auth.require_token)):
        tick = app.state.adapter.get_tick(symbol)
        if tick is None:
            raise HTTPException(status_code=404, detail=f"no price for {symbol}")
        return PriceResponse(
            symbol=tick.symbol, bid=tick.bid, ask=tick.ask, time=tick.time,
            source=DataSource.BROKER,
        )

    @app.get(f"{API_PREFIX}/rates/{{symbol}}", response_model=RatesResponse)
    def rates(
        symbol: str,
        timeframe: str = Query(default="M15"),
        count: int = Query(default=200, ge=1, le=5000),
        drop_forming: bool = Query(default=True),
        _: None = Depends(auth.require_token),
    ):
        from core.mt5.types import Timeframe

        tf = Timeframe.__members__.get(timeframe.upper())
        if tf is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown timeframe {timeframe!r}; "
                       f"expected one of {sorted(Timeframe.__members__)}",
            )
        df = app.state.adapter.get_candles(symbol, tf, count, drop_forming=drop_forming)
        if df is None or len(df) == 0:
            raise HTTPException(status_code=404, detail=f"no rates for {symbol}")
        return RatesResponse(
            symbol=symbol,
            timeframe=timeframe.upper(),
            time=[t.to_pydatetime() for t in df.index],
            open=[float(v) for v in df["open"]],
            high=[float(v) for v in df["high"]],
            low=[float(v) for v in df["low"]],
            close=[float(v) for v in df["close"]],
            tick_volume=[float(v) for v in df["tick_volume"]] if "tick_volume" in df else [],
            source=DataSource.BROKER,
        )

    # ── Positions / deals ───────────────────────────────────────────────────

    @app.get(f"{API_PREFIX}/positions", response_model=PositionsResponse)
    def positions(
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
        _: None = Depends(auth.require_token),
    ):
        rows = app.state.adapter.get_positions(symbol=symbol, magic=magic)
        return PositionsResponse(
            positions=[
                PositionResponse(
                    ticket=p.ticket, symbol=p.symbol, type=p.type, volume=p.volume,
                    price_open=p.price_open, price_current=p.price_current,
                    sl=p.sl, tp=p.tp, profit=p.profit, swap=p.swap,
                    magic=p.magic, time=p.time,
                )
                for p in rows
            ],
            count=len(rows),
            source=DataSource.BROKER,
        )

    @app.get(f"{API_PREFIX}/deals", response_model=DealsResponse)
    def deals(
        history_days: int = Query(default=7, ge=1, le=365),
        _: None = Depends(auth.require_token),
    ):
        """Recent deals. Returns an empty list rather than inventing data."""
        history = getattr(app.state.adapter, "get_deals", None)
        if not callable(history):
            return DealsResponse(deals=[], count=0, source=DataSource.UNAVAILABLE)
        rows = history(days=history_days)
        return DealsResponse(
            deals=[
                DealResponse(
                    ticket=d.ticket, symbol=d.symbol, type=d.type, volume=d.volume,
                    price=d.price, profit=d.profit, commission=d.commission,
                    swap=d.swap, magic=d.magic, position_id=d.position_id, time=d.time,
                )
                for d in rows
            ],
            count=len(rows),
            source=DataSource.BROKER,
        )

    # ── Order validation (read-only) ────────────────────────────────────────

    @app.post(f"{API_PREFIX}/order/validate", response_model=OrderValidationResponse)
    def validate_order(
        payload: OrderRequestModel,
        _: None = Depends(auth.require_token),
    ):
        """Validate against broker constraints. Never submits an order."""
        reasons: list[str] = []
        info = app.state.adapter.get_symbol_info(payload.symbol)
        if info is None:
            return OrderValidationResponse(
                valid=False, reasons=[f"unknown symbol {payload.symbol}"]
            )

        if info.volume_min is not None and payload.volume < info.volume_min:
            reasons.append(f"volume below broker minimum {info.volume_min}")
        if info.volume_max is not None and payload.volume > info.volume_max:
            reasons.append(f"volume above broker maximum {info.volume_max}")
        if info.volume_step:
            steps = round(payload.volume / info.volume_step)
            if abs(payload.volume - steps * info.volume_step) > 1e-6:
                reasons.append(f"volume must be a multiple of {info.volume_step}")

        min_stop = None
        if info.trade_stops_level is not None and info.point is not None:
            min_stop = info.trade_stops_level * info.point
            tick = app.state.adapter.get_tick(payload.symbol)
            if tick is not None:
                ref = tick.ask if payload.side == 0 else tick.bid
                for label, level in (("sl", payload.sl), ("tp", payload.tp)):
                    if level and abs(ref - level) < min_stop:
                        reasons.append(
                            f"{label} is within the broker minimum stop distance {min_stop}"
                        )

        return OrderValidationResponse(
            valid=not reasons,
            reasons=reasons,
            min_volume=info.volume_min,
            max_volume=info.volume_max,
            volume_step=info.volume_step,
            min_stop_distance=min_stop,
            stops_level=info.trade_stops_level,
        )

    # ── Writes (disabled by default) ────────────────────────────────────────

    @app.post(f"{API_PREFIX}/order", response_model=OrderResponse)
    def submit_order(payload: OrderRequestModel, _: None = Depends(auth.require_token)):
        require_writes()
        require_expected_account()

        if app.state.idempotency.seen(payload.idempotency_key):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"idempotency key {payload.idempotency_key!r} was already "
                       f"used; refusing to place a duplicate order",
            )

        validation = validate_order(payload, _)
        if not validation.valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="; ".join(validation.reasons),
            )

        logger.warning(
            "SUBMIT order %s %s %.2f (key=%s)",
            payload.symbol, "BUY" if payload.side == 0 else "SELL",
            payload.volume, payload.idempotency_key,
        )

        request = OrderRequest(
            symbol=payload.symbol,
            action=TradeAction.DEAL,
            type=OrderType.BUY if payload.side == 0 else OrderType.SELL,
            volume=payload.volume,
            price=0.0,
            sl=payload.sl,
            tp=payload.tp,
            deviation=payload.deviation,
            magic=payload.magic,
            comment=payload.comment,
        )
        result = app.state.adapter.send_order(request)
        # Unknown outcomes stay unknown; the caller must reconcile.
        state = state_from_retcode(result.retcode, result.volume, payload.volume)
        return OrderResponse(
            state=state.value,
            idempotency_key=payload.idempotency_key,
            retcode=result.retcode,
            deal=result.deal,
            order=result.order,
            volume=result.volume,
            price=result.price,
            comment=result.comment,
        )

    @app.post(f"{API_PREFIX}/position/modify", response_model=ActionResponse)
    def modify_position(
        payload: ModifyPositionRequest,
        _: None = Depends(auth.require_token),
    ):
        require_writes()
        require_expected_account()
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="position modification is not implemented on this bridge yet",
        )

    @app.post(f"{API_PREFIX}/position/close", response_model=ActionResponse)
    def close_position(
        payload: ClosePositionRequest,
        _: None = Depends(auth.require_token),
    ):
        require_writes()
        require_expected_account()
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="position closing is not implemented on this bridge yet",
        )

    return app


app = create_app()