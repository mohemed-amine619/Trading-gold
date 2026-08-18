"""
Strategy engine – Advanced Multi-Timeframe + AI Ensemble.

Signal scoring system (0 – 100 pts)
────────────────────────────────────
  Layer 1  AI ensemble             0–40 pts  (confidence × 40)
  Layer 2  Trend alignment         0–25 pts  (H1 + H4 EMA direction + ADX)
  Layer 3  Momentum confluence     0–20 pts  (RSI + MACD + Stochastic)
  Layer 4  Volume confirmation     0–10 pts  (above-average volume on bar)
  Layer 5  Market structure        0–5  pts  (room to run from recent HL)

Entry conditions (ALL must be true):
  ① score ≥ config.MIN_SIGNAL_SCORE
  ② ADX ≥ config.ADX_TRENDING_MIN   (avoids choppy markets)
  ③ session filter passes            (London / New York, optional)

Anti-repainting guarantee
--------------------------
  `prepare()` is only ever called on frames where the forming candle has
  already been removed by DataHandler (DataHandler.fetch_candles drops the
  last row).  All crossover/threshold comparisons therefore compare two
  *fully closed* bars → no lookahead, no repainting.

Backward compatibility
-----------------------
  `EmaRsiMultiTimeframeStrategy` is retained as the exported name so that
  existing GUI code and bot.py imports work without changes.  Its evaluate()
  signature gains an optional `h4_df` keyword argument (defaults to None).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pandas_ta as ta

from core.ai_model import AIEnsemble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    direction: int    = 0       # +1 long | -1 short | 0 flat / no trade
    reason: str       = ""
    close: float      = 0.0    # last closed-bar price (reference for entry)
    atr: float        = 0.0    # current ATR (used by RiskManager for SL/TP)
    score: float      = 0.0    # 0-100 confluence score
    ai_confidence: float = 0.0 # AI model directional confidence


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    def __init__(self, symbol: str, config):
        self.symbol          = symbol
        self.config          = config
        self.last_entry_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach indicator columns to the DataFrame (vectorized)."""

    @abstractmethod
    def evaluate(self, entry_df: pd.DataFrame, filter_df: pd.DataFrame,
                 h4_df: Optional[pd.DataFrame] = None) -> Signal:
        """Produce a Signal from closed candles only."""

    def snapshot(self) -> Signal:
        """Latest indicator values for trailing-stop / breakeven math."""
        if self.last_entry_df is None or self.last_entry_df.empty:
            return Signal()
        last = self.last_entry_df.iloc[-1]
        return Signal(0, close=float(last["close"]),
                      atr=float(last.get("atr", 0.0)))


# ---------------------------------------------------------------------------
# Advanced strategy implementation
# ---------------------------------------------------------------------------

class EmaRsiMultiTimeframeStrategy(BaseStrategy):
    """
    Advanced Multi-Timeframe AI Confluence Strategy.

    Replaces the legacy EMA-cross placeholder with a full scoring engine that
    combines an AI ensemble signal with multi-timeframe technical confluence.
    """

    def __init__(self, symbol: str, config):
        super().__init__(symbol, config)
        self._ai = (AIEnsemble(symbol, config)
                    if getattr(config, "AI_ENABLED", True) else None)

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df   = df.copy()
        c    = self.config
        cl   = df["close"]
        hi   = df["high"]
        lo   = df["low"]

        # Core trend indicators
        df["ema_fast"] = ta.ema(cl, length=c.EMA_FAST)
        df["ema_slow"] = ta.ema(cl, length=c.EMA_SLOW)
        df["ema20"]    = ta.ema(cl, length=20)
        df["ema100"]   = ta.ema(cl, length=100)

        # Shifted copies for crossover detection (two fully closed bars)
        df["ema_fast_prev"] = df["ema_fast"].shift(1)
        df["ema_slow_prev"] = df["ema_slow"].shift(1)

        # Oscillators
        df["rsi"]   = ta.rsi(cl, length=c.RSI_LENGTH)
        df["atr"]   = ta.atr(hi, lo, cl, length=c.ATR_PERIOD)

        # MACD
        try:
            macd = ta.macd(cl, fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                h = next((x for x in macd.columns if "MACDh" in x), None)
                s = next((x for x in macd.columns if "MACDs" in x), None)
                m = next((x for x in macd.columns if x.startswith("MACD_")), None)
                df["macd_hist"]   = macd[h] if h else 0.0
                df["macd_signal"] = macd[s] if s else 0.0
                df["macd_line"]   = macd[m] if m else 0.0
                df["macd_hist_prev"] = df["macd_hist"].shift(1)
        except Exception:
            df["macd_hist"] = df["macd_signal"] = df["macd_line"] = 0.0
            df["macd_hist_prev"] = 0.0

        # Stochastic
        try:
            stoch = ta.stoch(hi, lo, cl, k=14, d=3)
            if stoch is not None and not stoch.empty:
                k = next((x for x in stoch.columns if "STOCHk" in x), None)
                d = next((x for x in stoch.columns if "STOCHd" in x), None)
                df["stoch_k"] = stoch[k] if k else 50.0
                df["stoch_d"] = stoch[d] if d else 50.0
        except Exception:
            df["stoch_k"] = df["stoch_d"] = 50.0

        # ADX
        try:
            adx_df = ta.adx(hi, lo, cl, length=14)
            if adx_df is not None and not adx_df.empty:
                ac = next((x for x in adx_df.columns if x.startswith("ADX_")), None)
                pc = next((x for x in adx_df.columns if x.startswith("DMP_")), None)
                nc = next((x for x in adx_df.columns if x.startswith("DMN_")), None)
                df["adx"]      = adx_df[ac] if ac else 0.0
                df["di_plus"]  = adx_df[pc] if pc else 0.0
                df["di_minus"] = adx_df[nc] if nc else 0.0
        except Exception:
            df["adx"] = df["di_plus"] = df["di_minus"] = 0.0

        # Volume ratio
        df["vol_sma20"] = df["tick_volume"].rolling(20).mean()
        df["vol_ratio"] = df["tick_volume"] / (df["vol_sma20"] + 1e-10)

        return df

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _trend_score(self, df: Optional[pd.DataFrame], direction: int) -> float:
        """
        Return a 0-1 trend quality score for the given timeframe.
        Considers EMA alignment and RSI position.
        """
        if df is None or df.empty:
            return 0.0
        last = df.iloc[-1]
        ef   = last.get("ema_fast", float("nan"))
        es   = last.get("ema_slow", float("nan"))
        rsi  = last.get("rsi", 50.0)
        if pd.isna(ef) or pd.isna(es):
            return 0.0

        if direction == 1:
            ema_ok  = 1.0 if ef > es else 0.0
            rsi_ok  = min(1.0, max(0.0, (rsi - 45.0) / 20.0))
        else:
            ema_ok  = 1.0 if ef < es else 0.0
            rsi_ok  = min(1.0, max(0.0, (55.0 - rsi) / 20.0))

        return 0.65 * ema_ok + 0.35 * rsi_ok

    def _momentum_score(self, last: pd.Series, direction: int) -> float:
        """Return a 0-1 momentum score from RSI + MACD + Stochastic."""
        score = 0.0

        # RSI (weight 0.35)
        rsi = float(last.get("rsi", 50.0))
        if direction == 1:
            rsi_s = min(1.0, max(0.0, (rsi - 45.0) / 25.0))
        else:
            rsi_s = min(1.0, max(0.0, (55.0 - rsi) / 25.0))
        score += rsi_s * 0.35

        # MACD histogram (weight 0.35)
        mh      = float(last.get("macd_hist", 0.0))
        mh_prev = float(last.get("macd_hist_prev", 0.0))
        if direction == 1:
            macd_s = (1.0 if mh > 0 and mh > mh_prev else
                      0.5 if mh > 0 else 0.0)
        else:
            macd_s = (1.0 if mh < 0 and mh < mh_prev else
                      0.5 if mh < 0 else 0.0)
        score += macd_s * 0.35

        # Stochastic (weight 0.30)
        sk = float(last.get("stoch_k", 50.0))
        sd = float(last.get("stoch_d", 50.0))
        if direction == 1:
            stoch_s = (1.0 if sk > sd and 50 < sk < 80 else
                       0.5 if sk > sd else 0.0)
        else:
            stoch_s = (1.0 if sk < sd and 20 < sk < 50 else
                       0.5 if sk < sd else 0.0)
        score += stoch_s * 0.30

        return min(1.0, score)

    def _volume_score(self, last: pd.Series) -> float:
        """Return a 0-1 volume confirmation score."""
        vol_ratio = float(last.get("vol_ratio", 1.0))
        if pd.isna(vol_ratio):
            return 0.5
        # 0 at ratio=0.5, 1.0 at ratio=2.0+
        return min(1.0, max(0.0, (vol_ratio - 0.5) / 1.5))

    def _structure_score(self, df: pd.DataFrame, direction: int) -> float:
        """Return a 0-1 score based on room to run (distance from 20-bar HL)."""
        if len(df) < 20:
            return 0.5
        last  = df.iloc[-1]
        close = float(last["close"])
        atr   = float(last.get("atr", 0.0))
        if atr <= 0:
            return 0.5
        h20 = float(df["high"].rolling(20).max().iloc[-1])
        l20 = float(df["low"].rolling(20).min().iloc[-1])
        if direction == 1:
            room = (h20 - close) / atr   # ATR units of room to next resistance
        else:
            room = (close - l20) / atr   # ATR units of room to next support
        # 5+ ATR of room = perfect score
        return min(1.0, max(0.0, room / 5.0))

    def _session_ok(self) -> bool:
        """True if the current UTC hour falls inside a configured session."""
        if not getattr(self.config, "USE_SESSION_FILTER", True):
            return True
        now_h        = datetime.now(timezone.utc).hour
        lon_s        = getattr(self.config, "LONDON_START_UTC", 7)
        lon_e        = getattr(self.config, "LONDON_END_UTC",   16)
        ny_s         = getattr(self.config, "NY_START_UTC",     12)
        ny_e         = getattr(self.config, "NY_END_UTC",       21)
        return (lon_s <= now_h < lon_e) or (ny_s <= now_h < ny_e)

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(self, entry_df: pd.DataFrame, filter_df: pd.DataFrame,
                 h4_df: Optional[pd.DataFrame] = None) -> Signal:
        """
        Produce a trading signal by scoring all available evidence.

        Parameters
        ----------
        entry_df  : prepared M15 DataFrame (closed bars only)
        filter_df : prepared H1 DataFrame  (closed bars only)
        h4_df     : prepared H4 DataFrame  (optional, closed bars only)
        """
        self.last_entry_df = entry_df

        # --- basic sanity checks -----------------------------------------
        if entry_df is None or len(entry_df) < 3:
            return Signal(0, reason="insufficient data")
        if not self._session_ok():
            return Signal(0, reason="outside trading session (London/NY)")

        last  = entry_df.iloc[-1]
        close = float(last["close"])
        atr   = float(last.get("atr", 0.0))

        # --- ADX regime filter -------------------------------------------
        adx_val   = float(last.get("adx", 0.0))
        adx_min   = getattr(self.config, "ADX_TRENDING_MIN", 20)
        adx_strong = getattr(self.config, "ADX_STRONG_TREND", 30)
        if pd.notna(adx_val) and adx_val < adx_min:
            return Signal(0, reason=f"ADX={adx_val:.1f} < {adx_min} (ranging market, skip)")

        # --- AI update + prediction ---------------------------------------
        ai_direction = 0
        ai_conf      = 0.5
        if self._ai is not None:
            self._ai.update(entry_df)
            ai_sig       = self._ai.predict(entry_df)
            ai_direction = ai_sig.direction
            ai_conf      = ai_sig.confidence

        # --- Score both directions, take the higher one ------------------
        best_dir   = 0
        best_score = 0.0
        best_reason = "no confluence"

        for direction in [1, -1]:
            # Layer 1 – AI (0-40 pts)
            if ai_direction == direction:
                ai_pts = ai_conf * 40.0
            elif ai_direction == -direction:
                ai_pts = 0.0         # AI actively disagrees
            else:
                ai_pts = 12.0        # AI uncertain: partial credit

            # Layer 2 – Trend alignment (0-25 pts)
            h1_score = self._trend_score(filter_df, direction)   # 0-1
            h4_score = self._trend_score(h4_df,     direction)   # 0-1
            # H4 is the stronger signal (weight 13 vs 12)
            trend_pts = h1_score * 12.0 + h4_score * 13.0
            # ADX boost: strong trend → amplify trend score up to 25 cap
            if pd.notna(adx_val) and adx_val >= adx_strong:
                boost     = min(1.3, 1.0 + (adx_val - adx_strong) / 50.0)
                trend_pts = min(25.0, trend_pts * boost)

            # Layer 3 – Momentum (0-20 pts)
            mom_pts = self._momentum_score(last, direction) * 20.0

            # Layer 4 – Volume (0-10 pts)
            vol_pts = self._volume_score(last) * 10.0

            # Layer 5 – Structure (0-5 pts)
            struct_pts = self._structure_score(entry_df, direction) * 5.0

            total = ai_pts + trend_pts + mom_pts + vol_pts + struct_pts

            if total > best_score:
                best_score  = total
                best_dir    = direction
                best_reason = (
                    f"score={total:.1f}/100 "
                    f"[AI={ai_pts:.1f} trend={trend_pts:.1f} "
                    f"mom={mom_pts:.1f} vol={vol_pts:.1f} struct={struct_pts:.1f}] "
                    f"ADX={adx_val:.1f} ai_conf={ai_conf:.3f}"
                )

        min_score = getattr(self.config, "MIN_SIGNAL_SCORE", 55)
        if best_score < min_score:
            return Signal(0, reason=f"score={best_score:.1f} < threshold {min_score}")

        return Signal(
            direction     = best_dir,
            reason        = best_reason,
            close         = close,
            atr           = atr,
            score         = best_score,
            ai_confidence = ai_conf,
        )


# backward-compat alias
AdvancedAIStrategy = EmaRsiMultiTimeframeStrategy
