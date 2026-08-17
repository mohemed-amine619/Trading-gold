"""
DataHandler - fetches OHLCV history and produces clean pandas DataFrames.

Timezone normalization
----------------------
MT5 returns candle `time` as UNIX seconds *in the broker server timezone*
(the clock in the terminal's bottom-right corner). We label those timestamps
with the configured server offset and convert the index to UTC, so every
downstream calculation (new-bar detection, comparisons, logging) runs on one
consistent clock.

Closed-candle guarantee (no repainting)
---------------------------------------
`mt5.copy_rates_*` includes the candle currently being formed as its LAST row.
Signals computed on that row would change while the candle is still open
(repainting). We always fetch count + 1 rows and DROP the final row so
strategies only ever see fully closed candles.
"""
import logging
from dataclasses import dataclass
from datetime import timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

from core.mt5_engine import MT5Error

logger = logging.getLogger(__name__)

# MT5 timeframe enums are plain integers; this map bridges config strings to
# them. e.g. ENTRY_TIMEFRAME="M15" -> mt5.TIMEFRAME_M15.
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


@dataclass(frozen=True)
class SymbolInfo:
    """Snapshot of the static trading-relevant properties of a symbol."""
    name: str
    point: float              # smallest displayed price increment
    digits: int
    tick_size: float          # minimal price change used for trade calc
    tick_value: float         # account-ccy value of a one-tick move for 1.0 lot
    contract_size: float      # units per lot
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int          # min distance price <-> SL/TP in POINTS (0 = none)
    filling_mode: int         # bitmask of allowed ORDER_FILLING_* modes

    @classmethod
    def from_mt5(cls, info) -> "SymbolInfo":
        return cls(
            name=info.name,
            point=info.point,
            digits=info.digits,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            stops_level=info.trade_stops_level,
            filling_mode=info.filling_mode,
        )


class DataHandler:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        # The timezone object MT5 timestamps are expressed in (server wall clock).
        self._server_tz = timezone(timedelta(hours=config.SERVER_UTC_OFFSET_HOURS))

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        info = await self.engine.call(mt5.symbol_info, symbol)
        if info is None:
            code, desc = await self.engine.call(mt5.last_error)
            raise MT5Error(f"symbol_info({symbol}) failed: {desc} (code {code})")
        return SymbolInfo.from_mt5(info)

    async def fetch_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Return `count` CLOSED candles as a UTC-indexed DataFrame.

        Columns: open, high, low, close, tick_volume, spread.
        """
        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'")

        # Fetch one extra bar and drop it below -> guarantees closed candles.
        rates = await self.engine.call(mt5.copy_rates_from_pos, symbol, tf, 0, count + 1)
        if rates is None or len(rates) == 0:
            logger.warning("No rates for %s %s (last_error=%s)", symbol, timeframe,
                           await self.engine.call(mt5.last_error))
            return pd.DataFrame(columns=["open", "high", "low", "close",
                                         "tick_volume", "spread"])

        df = pd.DataFrame(rates).iloc[:-1].copy()  # drop the forming candle

        # Server-wall-clock seconds -> labelled server tz -> convert to UTC.
        # Equivalent math: utc_time = server_timestamp - offset_hours * 3600.
        df["time"] = (pd.to_datetime(df["time"], unit="s")
                      .dt.tz_localize(self._server_tz)
                      .dt.tz_convert("UTC"))
        df = df.set_index("time")
        return df[["open", "high", "low", "close", "tick_volume", "spread"]].astype(float)

    async def last_closed_bar_time(self, symbol: str, timeframe: str):
        """Timestamp of the most recently CLOSED bar, or None if unavailable.

        Used by the main loop to detect new bars cheaply (no full refetch).
        """
        df = await self.fetch_candles(symbol, timeframe, 2)
        if df.empty:
            return None
        return df.index[-1]
