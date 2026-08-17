"""
Strategy engine.

BaseStrategy - abstract contract every strategy must implement:
  * prepare(df):   enrich the raw OHLCV frame with indicator columns
                    (vectorized via pandas_ta on the whole frame)
  * evaluate(...): called once per NEW CLOSED bar, returns the desired action

Signal convention: direction in {+1 long, -1 short, 0 flat / no trade}.

Default implementation - EmaRsiMultiTimeframeStrategy (placeholder):
  Trend filter (higher timeframe, e.g. H1):
      EMA(50) > EMA(200) AND RSI(14) > 50   -> bullish regime
  Entry (entry timeframe, e.g. M15):
      EMA(50) crosses ABOVE EMA(200)        -> long
  Symmetric rules for shorts. SL/TP placement and sizing are delegated to the
  RiskManager (strategy only decides the DIRECTION and exposes ATR for stops).

Anti-repainting: `prepare()` never sees the forming candle (DataHandler drops
it), so the crossover at rows [-1]/[-2] compares two fully closed bars.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: int = 0        # +1 long, -1 short, 0 flat
    reason: str = ""
    close: float = 0.0        # last closed-bar price (reference for entry)
    atr: float = 0.0          # current ATR (used by RiskManager for SL/TP)


class BaseStrategy(ABC):
    """Interface all strategies must implement."""

    def __init__(self, symbol: str, config):
        self.symbol = symbol
        self.config = config
        self.last_entry_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach indicator columns to the DataFrame (vectorized)."""

    @abstractmethod
    def evaluate(self, entry_df: pd.DataFrame, filter_df: pd.DataFrame) -> Signal:
        """Produce a Signal from closed candles only."""

    def snapshot(self) -> Signal:
        """Latest indicator values (close + ATR) for trailing-stop math.

        Works even on ticks where no NEW bar appeared: the strategy re-reads
        its last prepared frame, which is always the most recent CLOSED bar.
        """
        if self.last_entry_df is None or self.last_entry_df.empty:
            return Signal()
        last = self.last_entry_df.iloc[-1]
        return Signal(0, close=float(last["close"]), atr=float(last["atr"]))


class EmaRsiMultiTimeframeStrategy(BaseStrategy):
    """Multi-timeframe trend follower: EMA crossover filtered by RSI regime."""

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = ta.ema(df["close"], length=self.config.EMA_FAST)
        df["ema_slow"] = ta.ema(df["close"], length=self.config.EMA_SLOW)
        df["rsi"] = ta.rsi(df["close"], length=self.config.RSI_LENGTH)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"],
                           length=self.config.ATR_PERIOD)
        # Shifted copies for crossover detection: compare the latest CLOSED bar
        # against the one before it. Both are closed -> deterministic signal.
        df["ema_fast_prev"] = df["ema_fast"].shift(1)
        df["ema_slow_prev"] = df["ema_slow"].shift(1)
        return df

    def _trend_filter(self, df: Optional[pd.DataFrame]) -> int:
        """+1 bullish, -1 bearish, 0 undecided (based on the highest CLOSED bar)."""
        if df is None or df.empty:
            return 0
        last = df.iloc[-1]
        if last["ema_fast"] > last["ema_slow"] and last["rsi"] > self.config.RSI_LONG_LEVEL:
            return 1
        if last["ema_fast"] < last["ema_slow"] and last["rsi"] < self.config.RSI_SHORT_LEVEL:
            return -1
        return 0

    def evaluate(self, entry_df: pd.DataFrame, filter_df: pd.DataFrame) -> Signal:
        self.last_entry_df = entry_df
        if entry_df is None or len(entry_df) < 3 or filter_df is None or filter_df.empty:
            return Signal(0, reason="insufficient data")

        last = entry_df.iloc[-1]
        prev = entry_df.iloc[-2]

        # Crossover on the last two CLOSED bars (no repainting possible).
        bull_cross = (last["ema_fast"] > last["ema_slow"]) and \
                     (prev["ema_fast"] <= prev["ema_slow"])
        bear_cross = (last["ema_fast"] < last["ema_slow"]) and \
                     (prev["ema_fast"] >= prev["ema_slow"])
        trend = self._trend_filter(filter_df)

        if bull_cross and trend == 1:
            return Signal(+1, reason="HTF bullish + EMA50/200 bullish cross",
                          close=float(last["close"]), atr=float(last["atr"]))
        if bear_cross and trend == -1:
            return Signal(-1, reason="HTF bearish + EMA50/200 bearish cross",
                          close=float(last["close"]), atr=float(last["atr"]))
        return Signal(0, reason="no crossover or filter disagreement")
