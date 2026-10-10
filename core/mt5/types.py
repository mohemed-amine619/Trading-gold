"""
core/mt5/types.py — Pydantic data-transfer objects for MT5 data.

These types decouple the rest of the system from raw MT5 structs so that:
  1. The codebase type-checks and validates cleanly.
  2. The mock adapter returns the same types as the real adapter.
  3. Tests can construct realistic data without MT5 installed.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums matching MT5 constants
# ─────────────────────────────────────────────────────────────────────────────

class Timeframe(IntEnum):
    M1  = 1
    M2  = 2
    M3  = 3
    M4  = 4
    M5  = 5
    M6  = 6
    M10 = 10
    M12 = 12
    M15 = 15
    M20 = 20
    M30 = 30
    H1  = 16385
    H2  = 16386
    H3  = 16387
    H4  = 16388
    H6  = 16390
    H8  = 16392
    H12 = 16396
    D1  = 16408
    W1  = 32769
    MN1 = 49153

    @classmethod
    def from_string(cls, s: str) -> "Timeframe":
        mapping = {
            "M1": cls.M1, "M5": cls.M5, "M15": cls.M15, "M30": cls.M30,
            "H1": cls.H1, "H4": cls.H4, "D1": cls.D1, "W1": cls.W1,
        }
        if s.upper() not in mapping:
            raise ValueError(f"Unknown timeframe string: {s}")
        return mapping[s.upper()]

    def to_string(self) -> str:
        mapping = {
            self.M1: "M1", self.M5: "M5", self.M15: "M15", self.M30: "M30",
            self.H1: "H1", self.H4: "H4", self.D1: "D1", self.W1: "W1",
        }
        return mapping.get(self, f"TF{int(self)}")


class OrderType(IntEnum):
    BUY             = 0
    SELL            = 1
    BUY_LIMIT       = 2
    SELL_LIMIT      = 3
    BUY_STOP        = 4
    SELL_STOP       = 5
    BUY_STOP_LIMIT  = 6
    SELL_STOP_LIMIT = 7
    CLOSE_BY        = 8


class TradeAction(IntEnum):
    DEAL    = 1   # Place market order
    PENDING = 5   # Place pending order
    SLTP    = 6   # Modify SL/TP
    MODIFY  = 7   # Modify pending order
    REMOVE  = 8   # Remove pending order
    CLOSE_BY = 10 # Close by opposite position


class RetCode(IntEnum):
    DONE               = 10009
    REQUOTE            = 10004
    PRICE_CHANGED      = 10015
    PRICE_OFF          = 10016
    REJECT             = 10006
    CANCEL             = 10007
    NO_MONEY           = 10019
    MARKET_CLOSED      = 10018
    TRADE_DISABLED     = 10017
    INVALID_STOPS      = 10022
    INVALID_VOLUME     = 10014
    CONNECTION         = 10035


# ─────────────────────────────────────────────────────────────────────────────
# Account info
# ─────────────────────────────────────────────────────────────────────────────

class AccountInfo(BaseModel):
    login: int
    server: str
    name: str = ""
    currency: str = "USD"
    leverage: int = 1
    balance: float
    equity: float
    margin: float = 0.0
    margin_free: float = 0.0
    margin_level: Optional[float] = None
    profit: float = 0.0
    credit: float = 0.0
    limit_orders: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Symbol info
# ─────────────────────────────────────────────────────────────────────────────

class SymbolInfo(BaseModel):
    name: str
    description: str = ""
    digits: int = 5
    point: float = 0.00001
    trade_tick_size: float = 0.00001
    trade_tick_value: float = 1.0
    trade_stops_level: int = 0
    trade_freeze_level: int = 0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    contract_size: float = 100000.0
    spread: int = 0
    spread_float: bool = True
    trade_mode: int = 4   # TRADE_MODE_FULL
    session_deals: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Tick
# ─────────────────────────────────────────────────────────────────────────────

class Tick(BaseModel):
    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Candle (OHLCV)
# ─────────────────────────────────────────────────────────────────────────────

class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0
    spread: float = 0.0
    real_volume: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────────────

class MT5Position(BaseModel):
    ticket: int
    symbol: str
    type: int          # 0 = BUY, 1 = SELL
    volume: float
    price_open: float
    price_current: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    magic: int = 0
    comment: str = ""
    time: int          # open time as Unix timestamp

    @property
    def is_buy(self) -> bool:
        return self.type == 0

    @property
    def direction(self) -> int:
        return 1 if self.is_buy else -1

    @property
    def total_pnl(self) -> float:
        return self.profit + self.swap + self.commission


# ─────────────────────────────────────────────────────────────────────────────
# Order send request / result
# ─────────────────────────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    symbol: str
    action: int = TradeAction.DEAL
    type: int = OrderType.BUY
    volume: float
    price: float
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 30
    magic: int = 888999
    comment: str = ""
    type_time: int = 0    # ORDER_TIME_GTC
    type_filling: int = 1 # ORDER_FILLING_IOC
    position: Optional[int] = None


class OrderResult(BaseModel):
    retcode: int
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0
    retcode_external: int = 0

    @property
    def success(self) -> bool:
        return self.retcode == RetCode.DONE

    @property
    def is_requote(self) -> bool:
        return self.retcode in (RetCode.REQUOTE, RetCode.PRICE_CHANGED, RetCode.PRICE_OFF)
