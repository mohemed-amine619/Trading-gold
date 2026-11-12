"""
core/ai/nn_features.py — leak-free features, labels and splits for the
sequence model. Pure numpy/pandas; no torch, no broker SDK.

Leakage rules this module enforces:

* every feature at bar ``t`` is a causal function of bars ``<= t`` (rolling and
  EWM windows only look backwards, nothing is centred or shifted forward);
* the scaler is fitted on the training rows only (the rows feeding the
  training windows, not the validation rows that pick the calibration and the
  threshold) and then applied unchanged to validation and live data
  (``FeatureScaler``);
* a label looks forward by ``horizon`` bars, so the last ``horizon`` rows have an
  unknown future and are marked ``UNKNOWN`` rather than guessed, and walk-forward
  folds leave a ``gap`` of at least ``horizon`` bars between train and test.

Classes: ``NOISE`` 0, ``LONG`` 1, ``SHORT`` 2.
  LONG  = the upper barrier (+barrier_mult x ATR) was touched first,
  SHORT = the lower barrier was touched first,
  NOISE = neither, or both inside one candle (their order is unknowable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from core.ai.labels import first_barrier_labels

NOISE, LONG, SHORT = 0, 1, 2
UNKNOWN = -1                     # future not yet observable; never train on it
N_CLASSES = 3
# Non-overlapping trades before an out-of-sample edge counts as measured. Lives
# here (torch-free) so the trainer's gate and the model share one number.
MIN_TRADES_FOR_EDGE = 30

FEATURE_NAMES = (
    "ret1", "ret3", "ret5",
    "atr_pct", "vol_regime",
    "dist_ema21", "dist_ema50",
    "rsi", "macd_hist",
    "range_pos",
    "body", "upper_wick", "lower_wick",
    "vol_ratio",
    "hour_sin", "hour_cos",
)

ATR_PERIOD = 14
_EPS = 1e-12


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder-style ATR from OHLC. Causal (EWM looks backwards only)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = gain / (loss + _EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary feature frame, same index as ``df``. Warm-up rows are NaN.

    Needs columns ``open, high, low, close`` and a DatetimeIndex (session
    encoding). ``tick_volume`` is optional; without it ``vol_ratio`` is 0.
    """
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df needs a DatetimeIndex for the session features")

    close = df["close"].astype(float)
    atr = _atr(df)
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()

    hi20 = df["high"].rolling(20, min_periods=20).max()
    lo20 = df["low"].rolling(20, min_periods=20).min()

    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)

    if "tick_volume" in df.columns:
        vol = df["tick_volume"].astype(float).clip(lower=0)
        vol_ratio = np.log((vol + 1.0) / (vol.rolling(20, min_periods=20).mean() + 1.0))
    else:
        vol_ratio = pd.Series(0.0, index=df.index)

    hours = df.index.hour + df.index.minute / 60.0
    angle = 2.0 * np.pi * hours / 24.0

    out = pd.DataFrame(index=df.index)
    out["ret1"] = np.log(close / close.shift(1))
    out["ret3"] = np.log(close / close.shift(3))
    out["ret5"] = np.log(close / close.shift(5))
    out["atr_pct"] = atr / close
    out["vol_regime"] = atr / atr.rolling(100, min_periods=50).mean() - 1.0
    out["dist_ema21"] = (close - ema21) / (atr + _EPS)
    out["dist_ema50"] = (close - ema50) / (atr + _EPS)
    out["rsi"] = (_rsi(close) - 50.0) / 50.0
    out["macd_hist"] = macd_hist / (atr + _EPS)
    out["range_pos"] = (close - lo20) / (hi20 - lo20 + _EPS) - 0.5
    out["body"] = (df["close"] - df["open"]) / (atr + _EPS)
    out["upper_wick"] = (df["high"] - body_hi) / (atr + _EPS)
    out["lower_wick"] = (body_lo - df["low"]) / (atr + _EPS)
    out["vol_ratio"] = vol_ratio
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out[list(FEATURE_NAMES)].replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, barrier_mult: float = 1.5, horizon: int = 24) -> np.ndarray:
    """Triple-barrier class per bar: NOISE / LONG / SHORT, or UNKNOWN.

    The last ``horizon`` bars and any bar whose ATR is not yet defined are
    UNKNOWN: their outcome cannot be known or was never defined, so they must
    be excluded from training, not treated as NOISE.
    """
    atr = _atr(df).to_numpy()
    first = first_barrier_labels(df["high"].to_numpy(), df["low"].to_numpy(),
                                 df["close"].to_numpy(), atr, barrier_mult, horizon)
    y = np.full(len(df), NOISE, dtype=int)
    y[first == 1.0] = LONG
    y[first == 0.0] = SHORT
    n = len(df)
    y[max(0, n - horizon):] = UNKNOWN
    y[~np.isfinite(atr) | (atr <= 0)] = UNKNOWN
    return y


@dataclass
class FeatureScaler:
    """Standardiser fitted on TRAINING rows only, then frozen.

    Statistics come from ``fit``; ``transform`` never updates them, so future
    data cannot leak into the scale of past windows. Output is clipped to
    ``+-clip`` standard deviations so a single spike cannot dominate.
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    clip: float = 5.0

    def fit(self, features: np.ndarray) -> "FeatureScaler":
        x = np.asarray(features, dtype=float)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) < 2:
            raise ValueError("need at least 2 finite rows to fit the scaler")
        self.mean = x.mean(axis=0)
        self.std = np.where(x.std(axis=0) > _EPS, x.std(axis=0), 1.0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler is not fitted")
        z = (np.asarray(features, dtype=float) - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(np.asarray(d["mean"], dtype=float), np.asarray(d["std"], dtype=float),
                   float(d.get("clip", 5.0)))


def window_end_indices(features: np.ndarray, labels: np.ndarray, window: int) -> np.ndarray:
    """The end bars of the windows ``make_windows`` would build, without
    building them: every bar ``e >= window-1`` whose label is known and whose
    ``window`` feature rows are all finite. Lets a caller split train /
    validation (and fit the scaler on the training part) before scaling."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if len(x) != len(y):
        raise ValueError("features and labels must have the same length")
    if window < 1 or len(x) < window:
        return np.empty((0,), dtype=int)
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(x).all(axis=1))])
    ends = np.arange(window - 1, len(x))
    ok = (bad[ends + 1] - bad[ends + 1 - window] == 0) & (y[ends] != UNKNOWN)
    return ends[ok].astype(int)


def make_windows(features: np.ndarray, labels: np.ndarray, window: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows ending at each bar.

    Returns ``(X, y, end_index)`` where ``X[k]`` holds rows
    ``end_index[k]-window+1 .. end_index[k]`` and ``y[k]`` is the label at the
    end bar. Windows containing a non-finite feature, or ending on an UNKNOWN
    label, are dropped (``window_end_indices``).
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    ends = window_end_indices(x, y, window)
    if len(ends) == 0:
        return (np.empty((0, window, x.shape[1])), np.empty((0,), dtype=int),
                np.empty((0,), dtype=int))
    xs = np.stack([x[end - window + 1:end + 1] for end in ends])
    return xs, np.asarray(y[ends], dtype=int), ends


def latest_window(features: np.ndarray, window: int) -> Optional[np.ndarray]:
    """The newest ``window`` rows, or ``None`` if short or not finite."""
    x = np.asarray(features, dtype=float)
    if len(x) < window:
        return None
    w = x[-window:]
    return w if np.isfinite(w).all() else None


def walk_forward_splits(n: int, train: int, test: int, gap: int, step: Optional[int] = None
                        ) -> Iterator[Tuple[slice, slice]]:
    """Rolling-origin folds: train, then ``gap`` bars skipped, then test.

    ``gap`` must be at least the label horizon, otherwise the last training
    labels were decided by prices that fall inside the test fold.
    """
    if min(train, test) < 1 or gap < 0:
        raise ValueError("train and test must be positive and gap non-negative")
    step = step or test
    start = 0
    while start + train + gap + test <= n:
        tr = slice(start, start + train)
        te = slice(start + train + gap, start + train + gap + test)
        yield tr, te
        start += step
