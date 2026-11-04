"""
core/ai/deep_features.py — inputs and targets for the deep two-branch network.

Pure numpy/pandas; no torch, no broker SDK.

Temporal branch input (``build_sequence_features``): the 16 stationary features
of ``nn_features.build_features`` plus
  * multi-horizon returns (4 / 16 / 48 bars — higher-timeframe proxies built from
    the one frame the bot already has, so no extra broker fetch);
  * EMA21 / EMA50 slopes, RSI and MACD-histogram derivatives;
  * a short-vs-long ATR ratio (volatility regime shift);
  * session flags: London, New York, and the London/NY overlap.
Everything is a causal function of bars ``<= t``. Sessions assume the frame's
index is UTC (what the MT5 export writes).

Structural branch input: ``core/ai/loc_features.py``.

Targets: the existing triple-barrier class (``nn_features.make_labels``) and an
"excursion multiplier": how far, in ATR, price travelled in the called
direction within the horizon (0 for NOISE). Barriers here are ATR multiples
because a structural stop only exists at decision time; the live gate still
anchors the real stop to ``structural_stop``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core.ai.loc_features import N_LOC_FEATURES, location_matrix
from core.ai.nn_features import (
    FEATURE_NAMES, LONG, SHORT, UNKNOWN, _atr, _rsi, build_features, make_labels,
)

EXTRA_NAMES = (
    "ret4", "ret16", "ret48",
    "ema21_slope", "ema50_slope",
    "rsi_delta", "macd_hist_delta",
    "atr_short_ratio",
    "sess_london", "sess_ny", "sess_overlap",
)
SEQ_FEATURE_NAMES = tuple(FEATURE_NAMES) + EXTRA_NAMES
N_SEQ_FEATURES = len(SEQ_FEATURE_NAMES)
EXCURSION_CAP = 6.0
_EPS = 1e-12


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary sequence feature frame (same index as ``df``; warm-up rows NaN)."""
    base = build_features(df)
    close = df["close"].astype(float)
    atr = _atr(df)
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - close.shift(1)).abs(),
                    (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    hour = df.index.hour

    out = base.copy()
    out["ret4"] = np.log(close / close.shift(4))
    out["ret16"] = np.log(close / close.shift(16))
    out["ret48"] = np.log(close / close.shift(48))
    out["ema21_slope"] = (ema21 - ema21.shift(3)) / (atr + _EPS)
    out["ema50_slope"] = (ema50 - ema50.shift(3)) / (atr + _EPS)
    out["rsi_delta"] = (_rsi(close) - _rsi(close).shift(3)) / 50.0
    out["macd_hist_delta"] = (macd_hist - macd_hist.shift(1)) / (atr + _EPS)
    out["atr_short_ratio"] = tr.rolling(5, min_periods=5).mean() / (atr + _EPS) - 1.0
    out["sess_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    out["sess_ny"] = ((hour >= 12) & (hour < 21)).astype(float)
    out["sess_overlap"] = ((hour >= 12) & (hour < 16)).astype(float)
    return out[list(SEQ_FEATURE_NAMES)].replace([np.inf, -np.inf], np.nan)


def make_excursion(df: pd.DataFrame, labels: np.ndarray, horizon: int) -> np.ndarray:
    """ATR-multiples travelled in the labelled direction within ``horizon`` bars.

    LONG: max(high[t+1..t+h]) - close[t]; SHORT: close[t] - min(low[t+1..t+h]);
    NOISE / UNKNOWN: 0. Clipped to ``[0, EXCURSION_CAP]``. Looks forward by
    design (it is a target); the last ``horizon`` rows are UNKNOWN in
    ``labels`` and therefore 0 and never trained on.
    """
    n = len(df)
    out = np.zeros(n, dtype=np.float64)
    if n <= horizon + 1:
        return out
    atr = _atr(df).to_numpy()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    fwd_hi = sliding_window_view(high[1:], horizon).max(axis=1)    # row t -> bars t+1..t+h
    fwd_lo = sliding_window_view(low[1:], horizon).min(axis=1)
    m = len(fwd_hi)                                                # = n - horizon
    t = np.arange(m)
    ok = np.isfinite(atr[:m]) & (atr[:m] > 0)
    lab = np.asarray(labels)[:m]
    up = np.where(ok, (fwd_hi - close[:m]) / np.where(ok, atr[:m], 1.0), 0.0)
    dn = np.where(ok, (close[:m] - fwd_lo) / np.where(ok, atr[:m], 1.0), 0.0)
    out[:m] = np.where(lab == LONG, up, np.where(lab == SHORT, dn, 0.0))
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, EXCURSION_CAP)


def make_windows_multi(seq: np.ndarray, loc: np.ndarray, labels: np.ndarray,
                       excursion: np.ndarray, window: int, stride: int = 1
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows ending at every ``stride``-th usable bar.

    Returns ``(X_seq, X_loc, y, exc, end_index)``. ``loc`` is the per-BAR
    location matrix (rows aligned to ``seq``); the window ending at ``e`` takes
    ``loc[e]``. Windows with a non-finite value or an UNKNOWN label are dropped.
    """
    seq = np.asarray(seq, dtype=float)
    loc = np.asarray(loc, dtype=float)
    y = np.asarray(labels)
    n = len(seq)
    if not (len(loc) == len(y) == len(excursion) == n):
        raise ValueError("seq, loc, labels and excursion must have the same length")
    finite = np.isfinite(seq).all(axis=1)
    ends = [e for e in range(window - 1, n, max(1, stride))
            if y[e] != UNKNOWN and finite[e - window + 1:e + 1].all()
            and np.isfinite(loc[e]).all()]
    if not ends:
        return (np.empty((0, window, seq.shape[1])), np.empty((0, loc.shape[1])),
                np.empty((0,), dtype=int), np.empty((0,)), np.empty((0,), dtype=int))
    ends_a = np.asarray(ends, dtype=int)
    xs = np.stack([seq[e - window + 1:e + 1] for e in ends])
    return xs, loc[ends_a], y[ends_a].astype(int), np.asarray(excursion)[ends_a], ends_a


def build_dataset(df: pd.DataFrame, window: int, horizon: int, barrier_mult: float,
                  stride: int = 1, loc_lookback: int = 200,
                  tick_size: Optional[float] = None) -> dict:
    """Everything the trainer needs, computed once over ``df``.

    Location rows are computed only for the bars that become window ends (every
    ``stride``-th usable bar), each from the bars up to and including it.
    """
    seq = build_sequence_features(df).to_numpy()
    labels = make_labels(df, barrier_mult, horizon)
    exc = make_excursion(df, labels, horizon)
    finite = np.isfinite(seq).all(axis=1)
    cand = [e for e in range(window - 1, len(df), max(1, stride))
            if labels[e] != UNKNOWN and finite[e - window + 1:e + 1].all()]
    loc = np.zeros((len(df), N_LOC_FEATURES), dtype=np.float64)
    if cand:
        loc[np.asarray(cand)] = location_matrix(df, cand, loc_lookback, tick_size)
    x_seq, x_loc, y, e, ends = make_windows_multi(seq, loc, labels, exc, window, stride)
    return {"seq": x_seq, "loc": x_loc, "y": y, "exc": e, "ends": ends, "seq_rows": seq}
