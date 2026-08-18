"""
AI Ensemble Model – per-symbol machine-learning signal generator.

Architecture
------------
Three complementary tree-based classifiers are combined via a soft-voting
ensemble:

  1. RandomForestClassifier   – robust, low-variance baseline
  2. GradientBoostingClassifier – sequential error correction
  3. ExtraTreesClassifier     – max variance reduction via random splits

All three are wrapped in a StandardScaler → VotingClassifier(soft) Pipeline.
The pipeline outputs class probabilities; we interpret P(up) > threshold as a
LONG signal and P(up) < (1 - threshold) as a SHORT signal.

Feature Engineering (~40 features)
------------------------------------
  • Price returns          – 1, 2, 3, 5, 10-bar pct changes
  • Candlestick structure  – body ratio, upper/lower wick, range/ATR
  • EMA distances          – close vs EMA(20/50/100/200), EMA spreads
  • MACD                   – histogram, delta, MACD vs signal line
  • RSI                    – value + 1-bar delta
  • Stochastic             – %K, %D, K-D spread
  • ADX                    – ADX value, +DI − −DI spread
  • Bollinger Bands        – %B, bandwidth
  • Volatility             – ATR % of price
  • Volume                 – tick-volume ratio vs 20-bar average
  • Market structure       – distance from 20-bar high/low as % of price
  • Momentum               – CCI(14), Williams %R(14)
  • Trend consistency      – fraction of up-bars in last 5 and 10 bars

Training / update policy
-------------------------
  * Rolling window of config.AI_TRAIN_BARS bars per symbol.
  * Retrains every config.AI_RETRAIN_BARS new bars.
  * Requires at least 150 samples and a non-degenerate label distribution
    (5% < up_rate < 95%) before fitting.
  * Target: 1 if next-bar close > current close, else 0.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column names (order matters – used for X matrix alignment)
# ---------------------------------------------------------------------------
FEATURE_COLS: List[str] = [
    # returns
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    # candlestick structure
    "body_ratio", "upper_wick", "lower_wick", "range_atr",
    # EMA positions
    "dist_ema20", "dist_ema50", "dist_ema100", "dist_ema200",
    "ema20_50_spread", "ema50_200_spread",
    # MACD
    "macd_hist", "macd_hist_delta", "macd_vs_signal",
    # RSI
    "rsi", "rsi_delta",
    # Stochastic
    "stoch_k", "stoch_d", "stoch_spread",
    # ADX
    "adx", "di_spread",
    # Bollinger Bands
    "bb_pct_b", "bb_width",
    # Volatility
    "atr_pct",
    # Volume
    "vol_ratio",
    # Market structure
    "dist_high20_pct", "dist_low20_pct",
    # Momentum
    "cci", "williams_r",
    # Trend consistency
    "up_bars_5", "up_bars_10",
]

_EP = 1e-10   # epsilon to avoid division-by-zero


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all feature columns to `df` (which must already contain an `atr`
    column from the strategy's prepare() call).  NaN in early rows is expected
    and will be handled by dropna() during training.
    """
    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]

    # ---- returns ----------------------------------------------------------
    for p in [1, 2, 3, 5, 10]:
        df[f"ret_{p}"] = close.pct_change(p)

    # ---- candlestick structure --------------------------------------------
    hl        = (high - low).clip(lower=_EP)
    body_top  = df[["open", "close"]].max(axis=1)
    body_bot  = df[["open", "close"]].min(axis=1)
    df["body_ratio"]  = (close - open_) / hl
    df["upper_wick"]  = (high - body_top) / hl
    df["lower_wick"]  = (body_bot - low) / hl
    atr_col           = df.get("atr", pd.Series(dtype=float))
    df["range_atr"]   = hl / (atr_col + _EP)

    # ---- EMA distances (normalized by EMA value) -------------------------
    for period in [20, 50, 100, 200]:
        ema = ta.ema(close, length=period)
        df[f"_ema{period}"] = ema
        df[f"dist_ema{period}"] = (close - ema) / (ema + _EP)
    df["ema20_50_spread"]  = (df["_ema20"]  - df["_ema50"])  / (df["_ema50"]  + _EP)
    df["ema50_200_spread"] = (df["_ema50"]  - df["_ema200"]) / (df["_ema200"] + _EP)

    # ---- MACD -------------------------------------------------------------
    try:
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            h_col  = next((c for c in macd_df.columns if "MACDh" in c), None)
            s_col  = next((c for c in macd_df.columns if "MACDs" in c), None)
            m_col  = next((c for c in macd_df.columns if c.startswith("MACD_")), None)
            if h_col:
                df["macd_hist"]       = macd_df[h_col]
                df["macd_hist_delta"] = df["macd_hist"] - df["macd_hist"].shift(1)
            if m_col and s_col:
                df["macd_vs_signal"]  = macd_df[m_col] - macd_df[s_col]
    except Exception:
        pass

    # ---- RSI --------------------------------------------------------------
    df["rsi"]       = ta.rsi(close, length=14)
    df["rsi_delta"] = df["rsi"] - df["rsi"].shift(1)

    # ---- Stochastic -------------------------------------------------------
    try:
        stoch_df = ta.stoch(high, low, close, k=14, d=3)
        if stoch_df is not None and not stoch_df.empty:
            k_col = next((c for c in stoch_df.columns if "STOCHk" in c), None)
            d_col = next((c for c in stoch_df.columns if "STOCHd" in c), None)
            if k_col:
                df["stoch_k"] = stoch_df[k_col]
            if d_col:
                df["stoch_d"] = stoch_df[d_col]
            df["stoch_spread"] = df.get("stoch_k", pd.Series(dtype=float)) \
                                 - df.get("stoch_d", pd.Series(dtype=float))
    except Exception:
        pass

    # ---- ADX --------------------------------------------------------------
    try:
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            adx_c  = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
            dmp_c  = next((c for c in adx_df.columns if c.startswith("DMP_")), None)
            dmn_c  = next((c for c in adx_df.columns if c.startswith("DMN_")), None)
            if adx_c:
                df["adx"] = adx_df[adx_c]
            if dmp_c and dmn_c:
                df["di_spread"] = adx_df[dmp_c] - adx_df[dmn_c]
    except Exception:
        pass

    # ---- Bollinger Bands --------------------------------------------------
    try:
        bb_df = ta.bbands(close, length=20, std=2.0)
        if bb_df is not None and not bb_df.empty:
            p_col = next((c for c in bb_df.columns if "BBP" in c), None)
            w_col = next((c for c in bb_df.columns if "BBB" in c), None)
            if p_col:
                df["bb_pct_b"] = bb_df[p_col]
            if w_col:
                df["bb_width"] = bb_df[w_col]
    except Exception:
        pass

    # ---- Volatility -------------------------------------------------------
    df["atr_pct"] = atr_col / (close + _EP)

    # ---- Volume -----------------------------------------------------------
    vol_sma       = df["tick_volume"].rolling(20).mean()
    df["vol_ratio"] = df["tick_volume"] / (vol_sma + _EP)

    # ---- Market structure -------------------------------------------------
    h20 = high.rolling(20).max()
    l20 = low.rolling(20).min()
    df["dist_high20_pct"] = (h20 - close) / (close + _EP)
    df["dist_low20_pct"]  = (close - l20) / (close + _EP)

    # ---- Momentum ---------------------------------------------------------
    try:
        df["cci"]       = ta.cci(high, low, close, length=14)
        df["williams_r"] = ta.willr(high, low, close, length=14)
    except Exception:
        pass

    # ---- Trend consistency ------------------------------------------------
    up_bar          = (close > close.shift(1)).astype(float)
    df["up_bars_5"]  = up_bar.rolling(5).sum()
    df["up_bars_10"] = up_bar.rolling(10).sum()

    return df


# ---------------------------------------------------------------------------
# Sklearn pipeline factory
# ---------------------------------------------------------------------------

def _make_pipeline() -> Pipeline:
    rf  = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=20,
        random_state=42, n_jobs=-1)
    gbt = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    et  = ExtraTreesClassifier(
        n_estimators=150, max_depth=8, min_samples_split=20,
        random_state=42, n_jobs=-1)
    voter = VotingClassifier(
        estimators=[("rf", rf), ("gbt", gbt), ("et", et)],
        voting="soft", n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", voter)])


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class AISignal:
    direction: int    = 0      # +1 long | -1 short | 0 uncertain
    confidence: float = 0.5   # directional probability (0.5 = coin flip)
    reason: str       = ""


# ---------------------------------------------------------------------------
# Per-symbol ensemble model
# ---------------------------------------------------------------------------

_MIN_TRAIN = 150   # absolute minimum training samples
_MIN_PRED  = 50    # minimum bars in input before prediction is attempted


class AIEnsemble:
    """
    Stateful per-symbol AI model.

    Lifecycle:
      • `update(df)` – called on every new closed bar; triggers a retrain
        when the bar counter reaches config.AI_RETRAIN_BARS.
      • `predict(df)` – returns an AISignal from the last bar of `df`.
    """

    def __init__(self, symbol: str, config):
        self.symbol         = symbol
        self.config         = config
        self._pipeline: Optional[Pipeline] = None
        self._feature_cols: List[str] = FEATURE_COLS[:]
        self._bars_since_train: int   = 0
        self._n_trains: int           = 0
        self._trained: bool           = False

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _build_xy(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray],
                                                    Optional[np.ndarray],
                                                    List[str]]:
        df = build_features(df)
        # Target: 1 if next bar closes higher, 0 otherwise.
        df["_target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        # The last row has no known target yet – drop it.
        df = df.iloc[:-1].copy()
        avail = [c for c in FEATURE_COLS if c in df.columns]
        df_clean = df[avail + ["_target"]].dropna()
        if len(df_clean) < _MIN_TRAIN:
            return None, None, avail
        return df_clean[avail].values, df_clean["_target"].values, avail

    def train(self, df: pd.DataFrame) -> bool:
        n      = min(len(df), self.config.AI_TRAIN_BARS)
        subset = df.iloc[-n:].copy()
        X, y, avail = self._build_xy(subset)
        if X is None:
            logger.warning("[AI] %s: need %d samples, have < %d – skipping train",
                           self.symbol, _MIN_TRAIN, _MIN_TRAIN)
            return False
        up_rate = y.mean()
        if up_rate < 0.05 or up_rate > 0.95:
            logger.warning("[AI] %s: degenerate labels (%.0f%% up) – skipping train",
                           self.symbol, up_rate * 100)
            return False
        try:
            pipe = _make_pipeline()
            pipe.fit(X, y)
            self._pipeline     = pipe
            self._feature_cols = avail
            self._trained      = True
            self._n_trains    += 1
            logger.info("[AI] %s trained (run #%d | samples=%d | up_rate=%.1f%%)",
                        self.symbol, self._n_trains, len(X), up_rate * 100)
            return True
        except Exception as exc:
            logger.error("[AI] %s training error: %s", self.symbol, exc)
            return False

    # ------------------------------------------------------------------
    # update (called every new closed bar)
    # ------------------------------------------------------------------
    def update(self, df: pd.DataFrame) -> None:
        self._bars_since_train += 1
        if not self._trained or self._bars_since_train >= self.config.AI_RETRAIN_BARS:
            if self.train(df):
                self._bars_since_train = 0

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> AISignal:
        if not self._trained or self._pipeline is None:
            return AISignal(0, 0.5, "model not trained yet")
        if len(df) < _MIN_PRED:
            return AISignal(0, 0.5, "not enough bars for prediction")
        try:
            feat_df = build_features(df)
            avail   = [c for c in self._feature_cols if c in feat_df.columns]
            row     = feat_df[avail].iloc[[-1]]
            if row.isnull().any(axis=1).values[0]:
                return AISignal(0, 0.5, "NaN in feature row")
            proba   = self._pipeline.predict_proba(row.values)[0]
            prob_up = float(proba[1])
            thr     = self.config.AI_CONFIDENCE_THRESHOLD
            if prob_up >= thr:
                return AISignal(+1, prob_up,
                                f"AI LONG  conf={prob_up:.3f}")
            if prob_up <= 1.0 - thr:
                conf = 1.0 - prob_up
                return AISignal(-1, conf,
                                f"AI SHORT conf={conf:.3f}")
            return AISignal(0, max(prob_up, 1.0 - prob_up), "AI uncertain")
        except Exception as exc:
            logger.error("[AI] %s predict error: %s", self.symbol, exc)
            return AISignal(0, 0.5, f"predict error: {exc}")
