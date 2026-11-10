"""
core/ai/deep_layout.py — torch-free single-tensor assembly for the dual-stream net.

The dual-stream model keeps one tensor layout so the existing windowing,
scaling, calibration and walk-forward code is reused unchanged:

    x[n, t, :N_SEQ]   the sequential (time-series) features at step t
    x[n, t, N_SEQ:]   the market-location state at step t

The temporal branch reads ``x[:, :, :N_SEQ]`` over the whole window; the
structural branch reads ``x[:, -1, N_SEQ:]``, the location state at the bar
being decided. Live, only that newest location state exists, so it is
broadcast across the window; the model ignores the older steps' copies.

``make_excursion`` is the auxiliary regression target: how far price ran from
the close, in ATR, within the horizon. It looks forward by design (it is a
label) and is NaN where the future is not yet known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.ai.location_features import (
    N_STRUCT, STRUCT_FEATURE_NAMES, build_location_features, latest_location_features,
)
from core.ai.nn_features import FEATURE_NAMES, _atr, build_features

ARCH_GRU = "gru"
ARCH_DUAL = "dual"
N_SEQ = len(FEATURE_NAMES)
EXCURSION_CAP_ATR = 6.0


def feature_names(arch: str) -> list:
    if arch == ARCH_DUAL:
        return list(FEATURE_NAMES) + list(STRUCT_FEATURE_NAMES)
    return list(FEATURE_NAMES)


def build_model_features(df: pd.DataFrame, arch: str = ARCH_GRU, stride: int = 8) -> np.ndarray:
    """Training/evaluation matrix for ``df``: one row per bar."""
    base = build_features(df).to_numpy()
    if arch != ARCH_DUAL:
        return base
    return np.hstack([base, build_location_features(df, stride).to_numpy()])


def latest_model_window(df: pd.DataFrame, window: int, arch: str = ARCH_GRU):
    """The newest ``window`` rows as the model expects them, or ``None``."""
    base = build_features(df).to_numpy()
    if len(base) < window:
        return None
    w = base[-window:]
    if not np.isfinite(w).all():
        return None
    if arch != ARCH_DUAL:
        return w
    struct = latest_location_features(df)
    if not np.isfinite(struct).all():
        return None
    return np.hstack([w, np.tile(struct, (window, 1))])


def make_excursion(df: pd.DataFrame, horizon: int = 24) -> np.ndarray:
    """Largest move from the close in either direction within the next
    ``horizon`` bars, in ATR, capped. NaN when the horizon is incomplete or the
    ATR is undefined."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = _atr(df).to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(n - horizon):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        up = high[i + 1:i + 1 + horizon].max() - close[i]
        dn = close[i] - low[i + 1:i + 1 + horizon].min()
        out[i] = min(max(up, dn) / a, EXCURSION_CAP_ATR)
    return out


__all__ = ["ARCH_GRU", "ARCH_DUAL", "N_SEQ", "N_STRUCT", "feature_names",
           "build_model_features", "latest_model_window", "make_excursion"]
