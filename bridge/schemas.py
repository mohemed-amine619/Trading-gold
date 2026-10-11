"""
bridge/schemas.py — Typed request/response contracts for the MT5 bridge.

Versioned wire format shared by the Windows bridge and the Linux core.

Design rules:

* Every response states *where the data came from* (``source``), so the core
  can distinguish a real broker value from a cached or synthetic one.
* Absence is explicit (``None``), never a plausible default.
* Nothing here imports MetaTrader5 — the SDK is confined to the adapter layer
  so these schemas stay testable on Linux.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Provenance of a value, so callers never mistake synthetic for real."""

    BROKER = "broker"        # live from the terminal
    CACHED = "cached"        # previously fetched, may be stale
    UNAVAILABLE = "unavailable"


class TerminalState(str, Enum):
    NOT_CONNECTED = "not_connected"
    CONNECTED = "connected"
    LOGGED_IN = "logged_in"
    ACCOUNT_MISMATCH = "account_mismatch"
    ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# Health / readiness
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Liveness: the bridge process is running. Says nothing about the terminal."""

    status: str = "ok"
    service: str = "mt5-bridge"
    version: str


class ReadyResponse(BaseModel):
    """Readiness: terminal connected AND authenticated to the expected account."""

    ready: bool
    terminal_state: TerminalState
    detail: str = ""
    # Populated only when the bridge is ready; never included otherwise.
    login: Optional[int] = None
    server: Optional[str] = None


class TerminalStatus(BaseModel):
    terminal_state: TerminalState
    login: Optional[int] = None
    server: Optional[str] = None
    account_name: Optional[str] = None
    last_error: Optional[str] = None
    source: DataSource = DataSource.BROKER


# ─────────────────────────────────────────────────────────────────────────────
# Account
# ─────────────────────────────────────────────────────────────────────────────

class AccountInfoResponse(BaseModel):
    login: Optional[int] = None
    server: Optional[str] = None
    currency: Optional[str] = None
    leverage: Optional[int] = None
    balance: Optional[float] = None
    equity: Optional[float] = None
    margin: Optional[float] = None
    margin_free: Optional[float] = None
    profit: Optional[float] = None
    source: DataSource = DataSource.BROKER


# ─────────────────────────────────────────────────────────────────────────────
# Instruments
# ─────────────────────────────────────────────────────────────────────────────

class SymbolInfoResponse(BaseModel):
    """Instrument specification.

    Optional numeric fields are ``None`` when the broker did not supply them.
    The core treats a None as "not tradeable" rather than substituting a
    default — see ``core.instruments.InstrumentSpec``.
    """

    name: str
    description: str = ""
    digits: Optional[int] = None
    point: Optional[float] = None
    trade_tick_size: Optional[float] = None
    trade_tick_value: Optional[float] = None
    trade_stops_level: Optional[int] = None
    trade_freeze_level: Optional[int] = None
    volume_min: Optional[float] = None
    volume_max: Optional[float] = None
    volume_step: Optional[float] = None
    contract_size: Optional[float] = None
    spread: Optional[float] = None
    trade_mode: Optional[int] = None
    source: DataSource = DataSource.BROKER


class SymbolsResponse(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    count: int = 0
    source: DataSource = DataSource.BROKER


# ─────────────────────────────────────────────────────────────────────────────
# Market data
# ─────────────────────────────────────────────────────────────────────────────

class PriceResponse(BaseModel):
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    time: Optional[datetime] = None
    source: DataSource = DataSource.BROKER


class RatesResponse(BaseModel):
    symbol: str
    timeframe: str
    # Column-oriented candles, matching core/mt5/adapter.get_candles().
    time: list[datetime] = Field(default_factory=list)
    open: list[float] = Field(default_factory=list)
    high: list[float] = Field(default_factory=list)
    low: list[float] = Field(default_factory=list)
    close: list[float] = Field(default_factory=list)
    tick_volume: list[float] = Field(default_factory=list)
    source: DataSource = DataSource.BROKER


# ─────────────────────────────────────────────────────────────────────────────
# Positions / orders / deals
# ─────────────────────────────────────────────────────────────────────────────

class PositionResponse(BaseModel):
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    magic: int = 0
    time: int = 0
    source: DataSource = DataSource.BROKER


class PositionsResponse(BaseModel):
    positions: list[PositionResponse] = Field(default_factory=list)
    count: int = 0
    source: DataSource = DataSource.BROKER


class DealResponse(BaseModel):
    ticket: int
    symbol: str
    type: int
    volume: float
    price: float
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    magic: int = 0
    position_id: int = 0
    time: int = 0
    source: DataSource = DataSource.BROKER


class DealsResponse(BaseModel):
    deals: list[DealResponse] = Field(default_factory=list)
    count: int = 0
    source: DataSource = DataSource.BROKER


# ─────────────────────────────────────────────────────────────────────────────
# Writes (disabled by default)
# ─────────────────────────────────────────────────────────────────────────────

class OrderRequestModel(BaseModel):
    """Order submission. Every field is validated before any broker call."""

    symbol: str = Field(min_length=1, max_length=32)
    side: int = Field(description="0=BUY, 1=SELL", ge=0, le=1)
    volume: float = Field(gt=0, le=1000)
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = Field(default=20, ge=0, le=1000)
    magic: int = Field(default=888999)
    comment: str = Field(default="", max_length=31)
    # Caller-supplied key. The bridge rejects repeats within its window so a
    # retried HTTP request cannot create a second order.
    idempotency_key: str = Field(min_length=1, max_length=128)


class OrderValidationResponse(BaseModel):
    """Dry-run validation. Never places an order."""

    valid: bool
    reasons: list[str] = Field(default_factory=list)
    # Present only when the broker supplied them.
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    volume_step: Optional[float] = None
    min_stop_distance: Optional[float] = None
    stops_level: Optional[int] = None


class OrderResponse(BaseModel):
    """
    Outcome of a submission.

    ``state`` is explicitly ``unknown`` when the terminal's answer was lost.
    Callers must reconcile in that case — it never means "rejected".
    """

    state: str = Field(
        description="filled | partially_filled | rejected | cancelled | not_sent | unknown"
    )
    idempotency_key: str
    retcode: int = 0
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    comment: str = ""


class ModifyPositionRequest(BaseModel):
    ticket: int = Field(gt=0)
    symbol: str = Field(min_length=1, max_length=32)
    sl: Optional[float] = None
    tp: Optional[float] = None


class ClosePositionRequest(BaseModel):
    ticket: int = Field(gt=0)
    symbol: str = Field(min_length=1, max_length=32)
    volume: Optional[float] = Field(default=None, gt=0)


class ActionResponse(BaseModel):
    status: str
    detail: str = ""
    # A modification is only "applied" once the broker confirms the new state.
    verified: bool = False
    sl: Optional[float] = None
    tp: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Uniform error shape so the core never parses prose."""

    error: str
    detail: str = ""
    # Machine-readable reason, e.g. "unauthenticated", "writes_disabled".
    code: str = "error"