"""
core/mt5/adapter.py — MT5 adapter interface + real + mock implementations.

The rest of the system ONLY uses the MT5Adapter ABC.
This means:
  - Unit tests run without MT5 installed (MockMT5Adapter)
  - Integration tests use a real or demo account (RealMT5Adapter)
  - The mock can be swapped with a paper-trade simulator in the future
"""

from __future__ import annotations

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from core.mt5.types import (
    AccountInfo,
    Candle,
    MT5Position,
    OrderRequest,
    OrderResult,
    RetCode,
    SymbolInfo,
    Tick,
    Timeframe,
    TradeAction,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class MT5Adapter(ABC):
    """
    Abstract contract for all MT5 operations.

    All methods that touch the broker are here.
    The trading engine never imports MetaTrader5 directly.
    """

    # ── Connection ──────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> bool:
        """Initialise MT5 connection and log in.  Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the terminal connection and account session are alive."""
        ...

    # ── Account ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        ...

    # ── Symbol ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        ...

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[Tick]:
        ...

    # ── Market data ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int,
        drop_forming: bool = True,
    ) -> pd.DataFrame:
        """
        Return `count` closed candles as a DataFrame with columns:
            time, open, high, low, close, tick_volume, spread
        Index: DatetimeIndex (UTC).
        If `drop_forming` is True (default), the still-forming bar is excluded.
        """
        ...

    # ── Positions ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> list[MT5Position]:
        ...

    # ── Orders ──────────────────────────────────────────────────────────────

    @abstractmethod
    def send_order(self, request: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def calc_margin(self, order_type: int, symbol: str, volume: float, price: float) -> float:
        """Return required margin for the given order in account currency."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Real MT5 adapter (Windows only)
# ─────────────────────────────────────────────────────────────────────────────

class RealMT5Adapter(MT5Adapter):
    """
    Wraps the MetaTrader5 Python package.

    This class is the ONLY place where `import MetaTrader5` appears.
    Everything else uses the abstract interface.
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str = "",
        timeout_ms: int = 120_000,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout_ms = timeout_ms
        self._mt5 = None  # lazy import

    def _import_mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError as e:
                raise RuntimeError(
                    "MetaTrader5 is not installed or not available on this platform. "
                    "Use MockMT5Adapter for testing."
                ) from e
        return self._mt5

    def connect(self) -> bool:
        mt5 = self._import_mt5()
        kwargs = {}
        if self._path:
            kwargs["path"] = self._path
        
        # Pass credentials to initialize to prevent it from failing if terminal auto-logs into broken account
        kwargs["login"] = self._login
        kwargs["password"] = self._password
        kwargs["server"] = self._server
        
        logger.info("Initialising MT5 terminal…")
        if not mt5.initialize(**kwargs, timeout=self._timeout_ms):
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False
        ok = mt5.login(self._login, password=self._password, server=self._server)
        if not ok:
            logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return False
        acc = mt5.account_info()
        if acc:
            logger.info(
                "Connected to %s (login=%d | balance=%.2f | equity=%.2f)",
                self._server, self._login, acc.balance, acc.equity,
            )
        return True

    def disconnect(self) -> None:
        mt5 = self._import_mt5()
        mt5.shutdown()
        logger.info("MT5 connection closed.")

    def is_connected(self) -> bool:
        try:
            mt5 = self._import_mt5()
            return mt5.terminal_info() is not None and mt5.account_info() is not None
        except Exception:
            return False

    def get_account_info(self) -> Optional[AccountInfo]:
        mt5 = self._import_mt5()
        acc = mt5.account_info()
        if acc is None:
            return None
        return AccountInfo(
            login=acc.login,
            server=acc.server,
            name=getattr(acc, "name", ""),
            currency=acc.currency,
            leverage=acc.leverage,
            balance=acc.balance,
            equity=acc.equity,
            margin=acc.margin,
            margin_free=acc.margin_free,
            margin_level=getattr(acc, "margin_level", None),
            profit=acc.profit,
            credit=getattr(acc, "credit", 0.0),
            limit_orders=getattr(acc, "limit_orders", 0),
        )

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        mt5 = self._import_mt5()
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return SymbolInfo(
            name=info.name,
            description=getattr(info, "description", ""),
            digits=info.digits,
            point=info.point,
            trade_tick_size=info.trade_tick_size,
            trade_tick_value=info.trade_tick_value,
            trade_stops_level=info.trade_stops_level,
            trade_freeze_level=getattr(info, "trade_freeze_level", 0),
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            contract_size=info.contract_size,
            spread=info.spread,
            spread_float=bool(info.spread_float),
            trade_mode=info.trade_mode,
        )

    def get_tick(self, symbol: str) -> Optional[Tick]:
        mt5 = self._import_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return Tick(
            symbol=symbol,
            time=datetime.fromtimestamp(tick.time, tz=timezone.utc),
            bid=tick.bid,
            ask=tick.ask,
            last=getattr(tick, "last", 0.0),
            volume=float(getattr(tick, "volume", 0)),
        )

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int,
        drop_forming: bool = True,
    ) -> pd.DataFrame:
        mt5 = self._import_mt5()
        fetch_count = count + 1 if drop_forming else count
        rates = mt5.copy_rates_from_pos(symbol, int(timeframe), 0, fetch_count)
        if rates is None or len(rates) == 0:
            logger.warning("get_candles: no data for %s tf=%s", symbol, timeframe)
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        if drop_forming and len(df) > 0:
            df = df.iloc[:-1]
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        cols = [c for c in ["open", "high", "low", "close", "tick_volume", "spread"] if c in df.columns]
        return df[cols].astype(float)

    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> list[MT5Position]:
        mt5 = self._import_mt5()
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol
        if magic is not None:
            kwargs["magic"] = magic
        raw = mt5.positions_get(**kwargs) or []
        result = []
        for p in raw:
            result.append(MT5Position(
                ticket=p.ticket,
                symbol=p.symbol,
                type=p.type,
                volume=p.volume,
                price_open=p.price_open,
                price_current=p.price_current,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                swap=p.swap,
                commission=getattr(p, "commission", 0.0),
                magic=p.magic,
                comment=p.comment,
                time=int(p.time),
            ))
        return result

    def send_order(self, request: OrderRequest) -> OrderResult:
        mt5 = self._import_mt5()
        req_dict = {
            "action":       request.action,
            "symbol":       request.symbol,
            "volume":       float(request.volume),
            "type":         request.type,
            "price":        float(request.price),
            "sl":           float(request.sl),
            "tp":           float(request.tp),
            "deviation":    request.deviation,
            "magic":        request.magic,
            "comment":      request.comment[:31],  # MT5 max comment length
            "type_time":    request.type_time,
            "type_filling": request.type_filling,
        }
        result = mt5.order_send(req_dict)
        if result is None:
            return OrderResult(retcode=0, comment="order_send returned None")
        return OrderResult(
            retcode=result.retcode,
            deal=getattr(result, "deal", 0),
            order=getattr(result, "order", 0),
            volume=getattr(result, "volume", 0.0),
            price=getattr(result, "price", 0.0),
            bid=getattr(result, "bid", 0.0),
            ask=getattr(result, "ask", 0.0),
            comment=getattr(result, "comment", ""),
            request_id=getattr(result, "request_id", 0),
            retcode_external=getattr(result, "retcode_external", 0),
        )

    def calc_margin(self, order_type: int, symbol: str, volume: float, price: float) -> float:
        mt5 = self._import_mt5()
        margin = mt5.order_calc_margin(order_type, symbol, volume, price)
        return float(margin) if margin is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Mock adapter (no MT5 required — for tests and CI)
# ─────────────────────────────────────────────────────────────────────────────

class MockMT5Adapter(MT5Adapter):
    """
    Deterministic mock for unit testing.

    Generates synthetic OHLCV data from a configurable seed.
    Accepts orders and maintains an in-memory position list.

    Instrument specifications are modelled per asset class rather than
    returning one gold-shaped set of constants for every symbol. Returning
    identical contract_size/tick_value for all symbols silently disabled
    position sizing (every instrument floored to volume_min), so the mock now
    carries realistic values and a realistic reference price per asset.
    """

    # Per-asset contract specifications used by get_symbol_info(). These are
    # representative of typical broker values for a retail CFD/MT5 account.
    _SPEC_TABLE: dict[str, dict] = {
        "metal": {
            "digits": 2, "point": 0.01, "contract_size": 100.0,
            "tick_value": 1.0, "volume_min": 0.01, "volume_max": 50.0,
            "spread": 20,
        },
        "forex": {
            "digits": 5, "point": 0.00001, "contract_size": 100_000.0,
            "tick_value": 1.0, "volume_min": 0.01, "volume_max": 100.0,
            "spread": 10,
        },
        "crypto": {
            "digits": 2, "point": 0.01, "contract_size": 1.0,
            "tick_value": 1.0, "volume_min": 0.01, "volume_max": 10.0,
            "spread": 500,
        },
        "index": {
            "digits": 2, "point": 0.1, "contract_size": 1.0,
            "tick_value": 1.0, "volume_min": 0.1, "volume_max": 100.0,
            "spread": 30,
        },
        "energy": {
            "digits": 2, "point": 0.01, "contract_size": 1000.0,
            "tick_value": 1.0, "volume_min": 0.01, "volume_max": 50.0,
            "spread": 25,
        },
    }

    _DEFAULT_SPEC = _SPEC_TABLE["forex"]

    # Reference price per asset class, so candle/tick generation is sane for
    # non-gold instruments.
    _BASE_PRICE: dict[str, float] = {
        "metal": 2000.0,
        "forex": 1.1000,
        "crypto": 40000.0,
        "index": 40000.0,
        "energy": 80.0,
    }

    def __init__(
        self,
        login: int = 12345678,
        balance: float = 10_000.0,
        seed: int = 42,
        symbol: str = "XAUUSDm",
        base_price: Optional[float] = None,
    ) -> None:
        self._login = login
        self._balance = balance
        self._equity = balance
        self._seed = seed
        self._symbol = symbol
        self._base_price_override = base_price
        # Per-symbol reference prices so a multi-symbol mock stays coherent.
        self._base_prices: dict[str, float] = {}
        self._asset_of: dict[str, str] = {}
        # Symbols this mock is willing to describe. An unlisted symbol returns
        # None from get_symbol_info(), as a real broker would for an unknown one.
        self._known: set[str] = {symbol} if symbol else set()
        self._connected = False
        self._positions: list[MT5Position] = []
        self._next_ticket = 10000
        self._rng = random.Random(seed)
        self._classify_known_symbols()

    # ── Asset classification ────────────────────────────────────────────────

    def _classify(self, symbol: str) -> str:
        """Best-effort asset class for a symbol, used to select mock specs."""
        from core.instruments import canonical_symbol, infer_asset_class

        if symbol not in self._asset_of:
            asset, _ = infer_asset_class(canonical_symbol(symbol))
            self._asset_of[symbol] = asset.value
        return self._asset_of[symbol]

    def _classify_known_symbols(self) -> None:
        """Seed classification for the default symbol."""
        if self._symbol:
            self._classify(self._symbol)

    def add_symbol(self, symbol: str) -> None:
        """Register a symbol this mock should describe."""
        self._known.add(symbol)
        self._classify(symbol)

    def get_symbols(self) -> list[str]:
        """Dynamically discoverable symbol list, as a real adapter would."""
        return sorted(self._known)

    def base_price_for(self, symbol: str) -> float:
        """Reference price for a symbol, honouring any explicit override."""
        if self._base_price_override is not None:
            return self._base_price_override
        return self._BASE_PRICE.get(self._classify(symbol), self._BASE_PRICE["forex"])

    def connect(self) -> bool:
        self._connected = True
        logger.info("[MOCK] Connected (login=%d, balance=%.2f)", self._login, self._balance)
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[MOCK] Disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> Optional[AccountInfo]:
        if not self._connected:
            return None
        return AccountInfo(
            login=self._login,
            server="MockBroker",
            name="Test Account",
            currency="USD",
            leverage=100,
            balance=self._balance,
            equity=self._equity,
            margin=0.0,
            margin_free=self._equity,
            margin_level=None,
            profit=self._equity - self._balance,
        )

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Per-asset contract specification for this symbol.

        Returns None for a symbol the mock has not been told about, mirroring
        a real broker: an unrecognised symbol has no specification. Callers
        (registry, bridge validation) depend on this being None rather than a
        fabricated default.

        Recognised symbols are those passed to ``known_symbols`` or inferred
        from the default symbol; use :meth:`add_symbol` to register more.
        """
        if symbol not in self._known:
            return None
        spec = self._SPEC_TABLE.get(self._classify(symbol), self._DEFAULT_SPEC)
        return SymbolInfo(
            name=symbol,
            description=f"Mock {symbol} ({self._classify(symbol)})",
            digits=spec["digits"],
            point=spec["point"],
            trade_tick_size=spec["point"],
            trade_tick_value=spec["tick_value"],
            trade_stops_level=0,
            trade_freeze_level=0,
            volume_min=spec["volume_min"],
            volume_max=spec["volume_max"],
            volume_step=0.01,
            contract_size=spec["contract_size"],
            spread=spec["spread"],
            trade_mode=4,  # SYMBOL_TRADE_MODE_FULL
        )

    def get_tick(self, symbol: str) -> Optional[Tick]:
        if not self._connected:
            return None
        base = self.base_price_for(symbol)
        # Spread expressed in the instrument's own price units.
        point = self.get_symbol_info(symbol).point or 0.01
        spread = (self.get_symbol_info(symbol).spread or 20) * point
        mid = base + self._rng.gauss(0, base * 0.0002)
        return Tick(
            symbol=symbol,
            time=datetime.now(timezone.utc),
            bid=mid - spread / 2,
            ask=mid + spread / 2,
        )

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int,
        drop_forming: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV using a random walk."""
        rng = random.Random(self._seed)
        base = self.base_price_for(symbol)
        prices = [base]
        for _ in range(count * 4 + 10):
            prices.append(prices[-1] + rng.gauss(0, base * 0.0005))

        rows = []
        now = int(datetime.now(timezone.utc).timestamp())
        tf_seconds = {
            Timeframe.M1: 60, Timeframe.M5: 300, Timeframe.M15: 900,
            Timeframe.M30: 1800, Timeframe.H1: 3600, Timeframe.H4: 14400,
            Timeframe.D1: 86400,
        }.get(timeframe, 900)

        for i in range(count):
            base_i = i * 4
            segment = prices[base_i: base_i + 5]
            o = segment[0]
            c = segment[-1]
            h = max(segment) + abs(rng.gauss(0, base * 0.0002))
            lo = min(segment) - abs(rng.gauss(0, base * 0.0002))
            rows.append({
                "time": pd.Timestamp(
                    (now - (count - i) * tf_seconds) * 10**9, tz="UTC"
                ),
                "open": o, "high": h, "low": lo, "close": c,
                "tick_volume": float(rng.randint(100, 5000)),
                "spread": 2.0,
            })
        df = pd.DataFrame(rows).set_index("time")
        return df

    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> list[MT5Position]:
        result = self._positions
        if symbol:
            result = [p for p in result if p.symbol == symbol]
        if magic is not None:
            result = [p for p in result if p.magic == magic]
        return list(result)

    def send_order(self, request: OrderRequest) -> OrderResult:
        if not self._connected:
            return OrderResult(retcode=RetCode.CONNECTION, comment="not connected")

        if request.action == TradeAction.DEAL:
            # Simulate a market fill
            tick = self.get_tick(request.symbol)
            fill_price = tick.ask if request.type == 0 else tick.bid  # BUY/SELL

            ticket = self._next_ticket
            self._next_ticket += 1

            pos = MT5Position(
                ticket=ticket,
                symbol=request.symbol,
                type=request.type,
                volume=request.volume,
                price_open=fill_price,
                price_current=fill_price,
                sl=request.sl,
                tp=request.tp,
                profit=0.0,
                swap=0.0,
                magic=request.magic,
                comment=request.comment,
                time=int(time.time()),
            )
            self._positions.append(pos)
            logger.info(
                "[MOCK] Order filled: ticket=%d %s %s vol=%.2f @ %.5f",
                ticket,
                "BUY" if request.type == 0 else "SELL",
                request.symbol,
                request.volume,
                fill_price,
            )
            return OrderResult(
                retcode=RetCode.DONE,
                deal=ticket,
                order=ticket,
                volume=request.volume,
                price=fill_price,
            )

        elif request.action == TradeAction.SLTP:
            # Modify SL/TP
            for pos in self._positions:
                if pos.ticket == getattr(request, "position", -1):
                    pos.sl = request.sl
                    pos.tp = request.tp
                    break
            return OrderResult(retcode=RetCode.DONE)

        return OrderResult(retcode=RetCode.REJECT, comment="unsupported action in mock")

    def calc_margin(self, order_type: int, symbol: str, volume: float, price: float) -> float:
        info = self.get_symbol_info(symbol)
        if info is None:
            return 0.0
        acc = self.get_account_info()
        leverage = acc.leverage if acc else 100
        return (volume * info.contract_size * price) / leverage
