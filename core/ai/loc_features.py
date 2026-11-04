"""
core/ai/loc_features.py — turn a ``MarketLocation`` into a fixed-length vector.

Pure numpy; no torch, no broker SDK. This is the "structural branch" input of
the deep network: where the market is (range position), what its structure is
(trend, BOS / CHoCH), what is near (scored S/R zones) and what is resting
(liquidity pools).

Honesty rules, same as ``core/market/location.py``:

* Every distance is in ATR, never in price points, so one model sees gold,
  forex and indices on the same scale (multi-asset mandate).
* Absent is not zero. Each optional quantity has an explicit ``*_present`` flag.
  The accompanying distance is set to its far sentinel (1.0 after scaling) only
  so the vector stays finite; the flag is what carries the meaning.
* An UNDEFINED location yields a vector whose ``loc_valid`` is 0 and whose other
  entries are the neutral/absent encoding. It is never "neutral market".
* Causal: the vector depends only on the frame it was computed from, so a
  training row at bar ``t`` is built from bars ``<= t`` only
  (``location_matrix`` slices the frame to guarantee that).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from core.market.liquidity import LiquidityState
from core.market.location import LocationBias, MarketLocation, analyse_location
from core.market.structure import BreakKind, StructureBias

LOC_FEATURE_NAMES = (
    "loc_valid",
    "range_pos",            # position in the active range, -0.5 .. +0.5
    "range_height_atr",     # range height / 20 ATR, clipped to 0..1
    "trend_up",             # HH/HL
    "trend_down",           # LH/LL
    "trend_range",
    "structure_conf",
    "bos_signed",           # direction of the last BOS * recency (-1..+1)
    "choch_signed",         # direction of the last CHoCH * recency
    "bos_recent",           # a BOS within RECENT_BARS (0/1)
    "choch_recent",
    "sup_present",
    "sup_dist_atr",         # distance to nearest support / DIST_CAP_ATR, 0..1
    "sup_strength",
    "sup_touches",          # touches / 6, clipped
    "res_present",
    "res_dist_atr",
    "res_strength",
    "res_touches",
    "liq_above_present",
    "liq_above_dist_atr",
    "liq_above_members",    # members / 5, clipped
    "liq_below_present",
    "liq_below_dist_atr",
    "liq_below_members",
    "swept_above_recent",   # an above-pool was swept within RECENT_BARS
    "swept_below_recent",
)
N_LOC_FEATURES = len(LOC_FEATURE_NAMES)

DIST_CAP_ATR = 10.0     # distances beyond this read as "far"
RECENT_BARS = 10
RECENCY_TAU = 20.0
_IDX = {name: i for i, name in enumerate(LOC_FEATURE_NAMES)}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _dist_feature(distance: Optional[float], atr: Optional[float]) -> float:
    """Distance in ATR scaled to 0..1; far / unknown = 1.0 (flag says which)."""
    if distance is None or atr is None or atr <= 0 or not math.isfinite(distance):
        return 1.0
    return _clip01(abs(distance) / atr / DIST_CAP_ATR)


def absent_vector() -> np.ndarray:
    """The encoding of an UNDEFINED location: ``loc_valid`` 0, everything absent."""
    v = np.zeros(N_LOC_FEATURES, dtype=np.float64)
    for name in ("sup_dist_atr", "res_dist_atr", "liq_above_dist_atr", "liq_below_dist_atr"):
        v[_IDX[name]] = 1.0
    return v


def location_features(loc: Optional[MarketLocation], n_bars: int) -> np.ndarray:
    """Feature vector for one ``MarketLocation``.

    ``n_bars`` is the length of the frame the location was computed from; event
    and sweep indices are positions in that frame, so recency is
    ``n_bars - 1 - index``.
    """
    v = absent_vector()
    if loc is None or not loc.is_valid or not loc.has_range:
        return v
    atr = loc.atr if (loc.atr and loc.atr > 0) else None
    last = n_bars - 1
    v[_IDX["loc_valid"]] = 1.0
    if loc.range_position is not None:
        v[_IDX["range_pos"]] = float(min(1.0, max(0.0, loc.range_position))) - 0.5
    if atr is not None:
        v[_IDX["range_height_atr"]] = _clip01((loc.range_high - loc.range_low) / atr / 20.0)

    st = loc.structure
    v[_IDX["trend_up"]] = 1.0 if st.bias is StructureBias.BULLISH else 0.0
    v[_IDX["trend_down"]] = 1.0 if st.bias is StructureBias.BEARISH else 0.0
    v[_IDX["trend_range"]] = 1.0 if st.bias is StructureBias.RANGING else 0.0
    v[_IDX["structure_conf"]] = _clip01(st.confidence)

    for kind, signed, recent in ((BreakKind.BOS, "bos_signed", "bos_recent"),
                                 (BreakKind.CHOCH, "choch_signed", "choch_recent")):
        ev = next((e for e in reversed(st.events) if e.kind is kind), None)
        if ev is None:
            continue
        age = max(0, last - int(ev.break_index))
        v[_IDX[signed]] = float(np.sign(ev.direction)) * math.exp(-age / RECENCY_TAU)
        v[_IDX[recent]] = 1.0 if age <= RECENT_BARS else 0.0

    sup, res = loc.nearest_support(), loc.nearest_resistance()
    if sup is not None:
        v[_IDX["sup_present"]] = 1.0
        v[_IDX["sup_dist_atr"]] = _dist_feature(loc.price - sup.upper, atr)
        v[_IDX["sup_strength"]] = _clip01(sup.strength)
        v[_IDX["sup_touches"]] = _clip01(sup.touches / 6.0)
    if res is not None:
        v[_IDX["res_present"]] = 1.0
        v[_IDX["res_dist_atr"]] = _dist_feature(res.lower - loc.price, atr)
        v[_IDX["res_strength"]] = _clip01(res.strength)
        v[_IDX["res_touches"]] = _clip01(res.touches / 6.0)

    above, below = loc.target_above(), loc.target_below()
    if above is not None:
        v[_IDX["liq_above_present"]] = 1.0
        v[_IDX["liq_above_dist_atr"]] = _dist_feature(above.price - loc.price, atr)
        v[_IDX["liq_above_members"]] = _clip01(above.members / 5.0)
    if below is not None:
        v[_IDX["liq_below_present"]] = 1.0
        v[_IDX["liq_below_dist_atr"]] = _dist_feature(loc.price - below.price, atr)
        v[_IDX["liq_below_members"]] = _clip01(below.members / 5.0)

    def _recently_swept(pools) -> float:
        for p in pools:
            if p.state is not LiquidityState.RESTING and p.swept_index is not None \
                    and last - int(p.swept_index) <= RECENT_BARS:
                return 1.0
        return 0.0

    v[_IDX["swept_above_recent"]] = _recently_swept(loc.liquidity.swept_above(loc.price))
    v[_IDX["swept_below_recent"]] = _recently_swept(loc.liquidity.swept_below(loc.price))
    return v


def location_vector_from_frame(df: pd.DataFrame, lookback: int = 200,
                               tick_size: Optional[float] = None) -> np.ndarray:
    """Analyse the newest ``lookback`` bars of ``df`` and encode the result.

    Never raises: a frame the location engine cannot read gives the absent
    encoding, exactly as an UNDEFINED location does.
    """
    try:
        frame = df.tail(lookback)
        loc = analyse_location(frame, tick_size=tick_size)
        return location_features(loc, len(frame))
    except Exception:
        return absent_vector()


def location_matrix(df: pd.DataFrame, ends: Iterable[int], lookback: int = 200,
                    tick_size: Optional[float] = None) -> np.ndarray:
    """One location vector per position in ``ends``.

    Row ``k`` is computed from ``df.iloc[: ends[k] + 1]`` only (the newest
    ``lookback`` bars of it), so it cannot see any later bar.
    """
    ends = list(ends)
    out = np.empty((len(ends), N_LOC_FEATURES), dtype=np.float64)
    for k, end in enumerate(ends):
        lo = max(0, int(end) + 1 - lookback)
        out[k] = location_vector_from_frame(df.iloc[lo:int(end) + 1], lookback, tick_size)
    return out
