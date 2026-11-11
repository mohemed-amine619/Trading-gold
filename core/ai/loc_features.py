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
  entries are the neutral/absent encoding. It is never "neutral market". The
  same holds for a location the extractor cannot trust: a price or ATR that is
  not finite, or event indices that do not fit ``n_bars``.
* Causal: the vector depends only on the frame it was computed from, so a
  training row at bar ``t`` is built from bars ``<= t`` only
  (``location_matrix`` slices the frame to guarantee that).

Encoding notes that are easy to get wrong
-----------------------------------------
* ``StructureEvent.break_index`` and ``LiquidityPool.swept_index`` are POSITIONS
  in the frame handed to ``analyse_location`` (``frame.index[break_index] ==
  break_time``), not labels and not offsets from the end. Recency is therefore
  ``n_bars - 1 - index`` where ``n_bars`` is that frame's length.
* ``range_pos`` is the engine's clamped range position minus 0.5. Price outside
  the range saturates at the edge (+-0.5): a breakout of 1 ATR and one of 10 ATR
  read the same. That is the engine's definition (a breakout is not a deeper
  premium); the vector does not carry the excess.
* The three ``trend_*`` flags are one-hot. All three 0 with ``loc_valid`` 1 means
  the range exists but structure is UNDEFINED (fewer than two confirmed swings
  of each kind).
* ``bos_signed`` / ``choch_signed`` decay as ``exp(-age / RECENCY_TAU)``. "No such
  event among the engine's retained events" and "an event so old its weight is
  ~0" share the encoding 0 on purpose; ``*_recent`` is the sharp (0/1) version.
* ``swept_above_recent`` / ``swept_below_recent`` are keyed on the pool KIND, not
  on where price is now: "above" is a buy-side (HIGH) pool, "below" a sell-side
  (LOW) pool, and a sweep that happened within ``RECENT_BARS`` counts even if
  price has since traded through the pool's level.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from core.market.liquidity import LiquidityState
from core.market.location import MarketLocation, analyse_location
from core.market.structure import BreakKind, StructureBias

logger = logging.getLogger(__name__)

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
    "swept_above_recent",   # a buy-side (HIGH) pool was swept within RECENT_BARS
    "swept_below_recent",   # a sell-side (LOW) pool was swept within RECENT_BARS
)
N_LOC_FEATURES = len(LOC_FEATURE_NAMES)

DIST_CAP_ATR = 10.0     # distances beyond this read as "far"
RECENT_BARS = 10
RECENCY_TAU = 20.0
_IDX = {name: i for i, name in enumerate(LOC_FEATURE_NAMES)}
_DIST_NAMES = ("sup_dist_atr", "res_dist_atr", "liq_above_dist_atr", "liq_below_dist_atr")

# Failures swallowed by ``location_vector_from_frame`` are logged loudly for the
# first few, then quietly: a systematic failure must be visible, a per-row one
# must not flood a 20 000-row training run.
_MAX_LOUD_FAILURES = 3
_failures_seen = 0


def _finite(x) -> bool:
    """True for a real, finite number (None, NaN and +-inf are not)."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _clip01(x: float) -> float:
    """Clip to 0..1. NaN is propagated, not turned into 0 (NaN means unknown)."""
    x = float(x)
    if math.isnan(x):
        return x
    return float(min(1.0, max(0.0, x)))


def _dist_feature(distance: Optional[float], atr: Optional[float]) -> float:
    """Distance in ATR scaled to 0..1; far / unknown = 1.0 (flag says which)."""
    if distance is None or atr is None or atr <= 0 or not math.isfinite(distance):
        return 1.0
    return _clip01(abs(distance) / atr / DIST_CAP_ATR)


def absent_vector() -> np.ndarray:
    """The encoding of an UNDEFINED location: ``loc_valid`` 0, everything absent."""
    v = np.zeros(N_LOC_FEATURES, dtype=np.float64)
    for name in _DIST_NAMES:
        v[_IDX[name]] = 1.0
    return v


def _recently_swept(pools, buy_side: bool, last: int) -> float:
    """1.0 when a pool of the given side was swept within ``RECENT_BARS`` bars."""
    for p in pools:
        if bool(p.is_above) != buy_side:
            continue
        if p.state is LiquidityState.RESTING or p.swept_index is None:
            continue
        if last - int(p.swept_index) <= RECENT_BARS:
            return 1.0
    return 0.0


def location_features(loc: Optional[MarketLocation], n_bars: int) -> np.ndarray:
    """Feature vector for one ``MarketLocation``.

    ``n_bars`` is the length of the frame the location was computed from; event
    and sweep indices are positions in that frame, so recency is
    ``n_bars - 1 - index``. A location that is UNDEFINED, or that cannot be read
    safely (non-finite price / range position / ATR, indices beyond ``n_bars``,
    any non-finite entry), yields ``absent_vector()``.
    """
    v = absent_vector()
    if loc is None or not loc.is_valid or not loc.has_range:
        return v

    # The engine classifies a NaN price as DISCOUNT (every comparison against NaN
    # is False and the ladder falls through), so a NaN close would otherwise be
    # encoded as a *valid* location at the very bottom of the range.
    atr = loc.atr
    if not (_finite(loc.price) and _finite(loc.range_position) and _finite(atr)
            and float(atr) > 0):
        return v
    atr = float(atr)

    n_bars = int(n_bars)
    last = n_bars - 1
    st = loc.structure
    indices = [int(e.break_index) for e in st.events]
    indices += [int(p.swept_index) for p in loc.liquidity.pools if p.swept_index is not None]
    if last < 0 or any(i > last for i in indices):
        logger.warning("location_features: n_bars=%d does not fit the location's event "
                       "indices (max %s); refusing to encode recency", n_bars,
                       max(indices) if indices else None)
        return v

    v[_IDX["loc_valid"]] = 1.0
    v[_IDX["range_pos"]] = _clip01(loc.range_position) - 0.5
    v[_IDX["range_height_atr"]] = _clip01((loc.range_high - loc.range_low) / atr / 20.0)

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

    pools = loc.liquidity.pools
    v[_IDX["swept_above_recent"]] = _recently_swept(pools, True, last)
    v[_IDX["swept_below_recent"]] = _recently_swept(pools, False, last)

    if not np.isfinite(v).all():
        return absent_vector()
    return v


def _check_lookback(lookback: int) -> int:
    lookback = int(lookback)
    if lookback < 1:
        # ``DataFrame.tail(0)`` is empty and ``tail(-k)`` drops the FIRST k rows,
        # neither of which is a lookback.
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    return lookback


def location_vector_from_frame(df: pd.DataFrame, lookback: int = 200,
                               tick_size: Optional[float] = None) -> np.ndarray:
    """Analyse the newest ``lookback`` bars of ``df`` and encode the result.

    Never raises on the frame: one the location engine cannot read gives the
    absent encoding, exactly as an UNDEFINED location does. (A non-positive
    ``lookback`` is a caller bug and does raise ``ValueError``.)
    """
    global _failures_seen
    lookback = _check_lookback(lookback)
    try:
        frame = df.tail(lookback)
        loc = analyse_location(frame, tick_size=tick_size)
        return location_features(loc, len(frame))
    except Exception as exc:        # contract: never raise on an unreadable frame
        _failures_seen += 1
        if _failures_seen <= _MAX_LOUD_FAILURES:
            logger.warning("location_vector_from_frame: unreadable frame, encoding as "
                           "absent (%s: %s)", type(exc).__name__, exc)
        else:
            logger.debug("location_vector_from_frame: unreadable frame (%r)", exc)
        return absent_vector()


def location_matrix(df: pd.DataFrame, ends: Iterable[int], lookback: int = 200,
                    tick_size: Optional[float] = None) -> np.ndarray:
    """One location vector per position in ``ends``.

    Row ``k`` is computed from ``df.iloc[: ends[k] + 1]`` only (the newest
    ``lookback`` bars of it), so it cannot see any later bar. ``ends`` are
    positions into ``df`` (``0 <= end < len(df)``); anything else raises
    ``ValueError`` rather than silently describing a different bar.
    """
    lookback = _check_lookback(lookback)
    positions = []
    for end in ends:
        pos = int(end)
        if pos != end or pos < 0 or pos >= len(df):
            raise ValueError(f"end {end!r} is not a position in a frame of {len(df)} bars")
        positions.append(pos)
    out = np.empty((len(positions), N_LOC_FEATURES), dtype=np.float64)
    for k, end in enumerate(positions):
        lo = max(0, end + 1 - lookback)
        out[k] = location_vector_from_frame(df.iloc[lo:end + 1], lookback, tick_size)
    return out
