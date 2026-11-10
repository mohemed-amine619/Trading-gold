"""
core/ai/location_features.py — the market-location state as a dense vector for
the neural network's structural branch.

Every distance is an ATR multiple (multi-asset mandate: never a fixed point
count) and every value is bounded, so the vector is stationary and comparable
across instruments.

A quantity that does not exist (no support zone, no confirmed range) is NOT
encoded as 0: 0 is a real distance. Each such quantity has a ``has_*`` flag and
a neutral fill, so the network can tell "none" from "right here".

``build_location_features`` is causal: row ``i`` is computed from bars
``<= i`` only. The full location read is expensive, so it is recomputed every
``stride`` bars and carried forward; a carried row is stale by at most
``stride - 1`` bars but never contains the future.

No MetaTrader5 import: this module stays Linux-testable.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from core.market.location import MarketLocation, analyse_location
from core.market.structure import StructureBias

STRUCT_FEATURE_NAMES = (
    "trend_state",          # +1 HH/HL, -1 LH/LL, 0 ranging or undefined
    "trend_confidence",
    "bos_signed",           # +/- exp(-age/DECAY) of the last BOS, 0 if none
    "choch_signed",         # same for the last CHoCH
    "has_range",
    "range_pos",            # position in the active range, centred (-0.5..0.5)
    "has_support",
    "support_dist_atr",
    "support_strength",
    "has_resistance",
    "resistance_dist_atr",
    "resistance_strength",
    "has_liq_above",
    "liq_above_dist_atr",
    "has_liq_below",
    "liq_below_dist_atr",
    "swept_above",          # any buy-side pool above price already taken
    "swept_below",
)
N_STRUCT = len(STRUCT_FEATURE_NAMES)

_DIST_CAP_ATR = 5.0
_EVENT_DECAY_BARS = 20.0
_LOOKBACK = 300
_MIN_BARS = 40


def _event_signal(event, n: int) -> float:
    if event is None:
        return 0.0
    age = max(0, (n - 1) - int(event.break_index))
    return float(event.direction) * math.exp(-age / _EVENT_DECAY_BARS)


def _dist(value: Optional[float], atr: Optional[float]) -> tuple[float, float]:
    """``(has, distance_in_atr)`` with the distance capped; (0, cap) when absent."""
    if value is None or atr is None or atr <= 0 or not math.isfinite(value):
        return 0.0, _DIST_CAP_ATR
    return 1.0, float(min(max(value / atr, 0.0), _DIST_CAP_ATR))


def location_feature_vector(loc: MarketLocation, n_bars: int) -> np.ndarray:
    """Dense vector for one location read; ``n_bars`` is the frame length used
    to age the structural events."""
    out = np.zeros(N_STRUCT, dtype=float)
    atr = loc.atr
    st = loc.structure

    if st.bias is StructureBias.BULLISH:
        out[0] = 1.0
    elif st.bias is StructureBias.BEARISH:
        out[0] = -1.0
    out[1] = float(min(max(st.confidence, 0.0), 1.0))
    out[2] = _event_signal(st.last_bos, n_bars)
    out[3] = _event_signal(st.last_choch, n_bars)

    if loc.has_range and loc.range_position is not None:
        out[4] = 1.0
        out[5] = float(min(max(loc.range_position, 0.0), 1.0)) - 0.5

    sup = loc.nearest_support()
    out[6], out[7] = _dist(loc.distance_to_support(), atr)
    out[8] = float(sup.strength) if sup is not None else 0.0
    res = loc.nearest_resistance()
    out[9], out[10] = _dist(loc.distance_to_resistance(), atr)
    out[11] = float(res.strength) if res is not None else 0.0

    above, below = loc.target_above(), loc.target_below()
    out[12], out[13] = _dist(None if above is None else above.price - loc.price, atr)
    out[14], out[15] = _dist(None if below is None else loc.price - below.price, atr)
    out[16] = 1.0 if loc.liquidity.swept_above(loc.price) else 0.0
    out[17] = 1.0 if loc.liquidity.swept_below(loc.price) else 0.0

    out[~np.isfinite(out)] = 0.0
    return out


def _neutral_row() -> np.ndarray:
    row = np.zeros(N_STRUCT, dtype=float)
    row[7] = row[10] = row[13] = row[15] = _DIST_CAP_ATR
    return row


def latest_location_features(df: pd.DataFrame) -> np.ndarray:
    """Structural vector at the newest bar of ``df`` (what the live bot sees)."""
    tail = df.tail(_LOOKBACK)
    if len(tail) < _MIN_BARS:
        return _neutral_row()
    return location_feature_vector(analyse_location(tail), len(tail))


def build_location_features(df: pd.DataFrame, stride: int = 8) -> pd.DataFrame:
    """Causal structural features for every bar of ``df`` (same index).

    Bars before ``_MIN_BARS`` are NaN (not enough history to read structure),
    so the window builder drops them rather than training on a guess.
    """
    stride = max(1, int(stride))
    n = len(df)
    values = np.full((n, N_STRUCT), np.nan, dtype=float)
    last: Optional[np.ndarray] = None
    for i in range(n):
        if i + 1 < _MIN_BARS:
            continue
        if last is None or (i % stride) == 0:
            tail = df.iloc[max(0, i + 1 - _LOOKBACK):i + 1]
            last = location_feature_vector(analyse_location(tail), len(tail))
        values[i] = last
    return pd.DataFrame(values, index=df.index, columns=list(STRUCT_FEATURE_NAMES))
