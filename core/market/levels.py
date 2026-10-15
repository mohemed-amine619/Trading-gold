"""
core/market/levels.py — Support and resistance as scored zones.

A single price is not a level. A *zone* is: a band of prices the market has
reacted from more than once, with a measurable history that says how much it
deserves to be trusted.

Two deliberate choices, both driven by the project's multi-asset mandate:

**Tolerance is ATR-derived, never a fixed point count.** Gold at 2400 with a
$6 ATR and EURUSD at 1.0850 with a 0.0008 ATR cannot share a hardcoded
"within 10 points" rule — one clusters every swing into a single blob, the
other never clusters anything. Every threshold here derives from the series'
own volatility, so an instrument we have never seen still behaves sanely.

**Rounding comes from the broker or not at all.** Levels are snapped to
``tick_size`` from ``InstrumentSpec`` when one is supplied. When it is not, the
level is returned unrounded rather than assuming two decimals, because
substituting a plausible default for absent broker metadata is exactly the
defect this repo already had to undo once (``core/instruments/model.py``).

Scoring is where a zone earns the right to be trusted. Strength combines how
often the market reacted, how recently, and — importantly — whether the zone
is *fresh*. A level the market has never returned to still holds untouched
order blocks; one that has been retested six times is largely spent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.market.swings import SwingKind, SwingPoint

from core.market.series import resolve_atr, valid_tick_size

logger = logging.getLogger(__name__)


class LevelKind(str, Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


@dataclass(frozen=True)
class LevelZone:
    """
    A band of price with a measured reaction history.

    Attributes:
        kind: ``SUPPORT`` or ``RESISTANCE`` *as currently respected*. A broken
            support becomes resistance, which is why ``kind`` is not simply the
            swing type that formed it — see ``formed_as``.
        price: Zone centre, snapped to tick size when known.
        lower / upper: Zone edges. A reaction is anything trading inside them.
        formed_as: The swing kind that originally built the zone.
        touches: Swings in the cluster, plus later bars that traded into it.
        strength: 0..1, see ``build_levels`` for the weighting.
        broken: True once a close traded decisively beyond the far edge.
        broken_index: Where that happened, for freshness maths.
    """

    kind: LevelKind
    price: float
    lower: float
    upper: float
    formed_as: SwingKind
    touches: int = 1
    strength: float = 0.0
    first_index: int = 0
    last_index: int = 0
    broken: bool = False
    broken_index: Optional[int] = None
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def is_resistance(self) -> bool:
        return self.kind is LevelKind.RESISTANCE

    @property
    def is_support(self) -> bool:
        return self.kind is LevelKind.SUPPORT

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "price": self.price,
            "lower": self.lower,
            "upper": self.upper,
            "formed_as": self.formed_as.value,
            "touches": self.touches,
            "strength": round(self.strength, 4),
            "broken": self.broken,
            "broken_index": self.broken_index,
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


@dataclass(frozen=True)
class LevelMap:
    """All zones for one symbol/timeframe, plus convenience lookups."""

    zones: List[LevelZone] = field(default_factory=list)
    atr: Optional[float] = None

    def above(self, price: float) -> List[LevelZone]:
        """Resistance zones strictly above ``price``, nearest first."""
        out = [z for z in self.zones if z.is_resistance and z.lower > price]
        return sorted(out, key=lambda z: z.lower)

    def below(self, price: float) -> List[LevelZone]:
        """Support zones strictly below ``price``, nearest first."""
        out = [z for z in self.zones if z.is_support and z.upper < price]
        return sorted(out, key=lambda z: z.upper, reverse=True)

    def nearest_above(self, price: float) -> Optional[LevelZone]:
        got = self.above(price)
        return got[0] if got else None

    def nearest_below(self, price: float) -> Optional[LevelZone]:
        got = self.below(price)
        return got[0] if got else None

    def strongest(self, kind: Optional[LevelKind] = None, limit: int = 5) -> List[LevelZone]:
        pool = [z for z in self.zones if kind is None or z.kind is kind]
        return sorted(pool, key=lambda z: z.strength, reverse=True)[:limit]

    def to_dict(self) -> dict:
        return {"atr": self.atr, "zones": [z.to_dict() for z in self.zones]}


def build_levels(
    df: pd.DataFrame,
    swings: Sequence[SwingPoint],
    atr: Optional[float] = None,
    tolerance_atr: float = 0.30,
    min_touches: int = 1,
    max_zones: int = 24,
    tick_size: Optional[float] = None,
    break_margin_atr: float = 0.10,
) -> LevelMap:
    """
    Cluster swing points into scored support/resistance zones.

    Args:
        df: OHLCV frame, used for touch counting and break detection.
        swings: Swings to cluster. Unconfirmed ones are dropped here; a level
            that can move cannot be a level.
        atr: Current ATR. When ``None`` it is derived from the frame's own
            true range, which is market data rather than a substituted default.
            An empty map is returned only when no scale can be established at
            all, in which case there is no principled tolerance and the caller
            must abstain rather than be handed an arbitrary clustering.
        tolerance_atr: Cluster width as a multiple of ATR. 0.30 means swings
            within ~0.3 ATR of each other are the same level.
        min_touches: Discard zones with fewer reactions than this.
        max_zones: Keep the strongest N, so the map stays bounded.
        tick_size: Broker minimum price increment. Used to snap level prices.
            ``None`` leaves prices unrounded, which is honest.
        break_margin_atr: How far beyond the zone edge a close must travel
            before the zone counts as broken.

    Returns:
        A ``LevelMap``. Empty when there is insufficient data or no ATR.
    """
    if df is None or df.empty:
        return LevelMap([], atr)

    confirmed = [s for s in swings if s.confirmed]
    if len(confirmed) < 2:
        return LevelMap([], atr)

    tick_size = valid_tick_size(tick_size)
    resolved_atr = atr if (atr is not None and atr > 0) else resolve_atr(df)
    if resolved_atr is None or resolved_atr <= 0:
        # Without volatility there is no scale-free notion of "close enough".
        # Returning an empty map forces the caller to abstain instead of
        # silently clustering with an arbitrary tolerance.
        logger.debug("build_levels: no usable ATR, refusing to cluster")
        return LevelMap([], None)

    tolerance = resolved_atr * tolerance_atr
    if tick_size and tick_size > 0:
        tolerance = max(tolerance, tick_size)

    highs = [s for s in confirmed if s.kind is SwingKind.HIGH]
    lows = [s for s in confirmed if s.kind is SwingKind.LOW]

    zones: List[LevelZone] = []
    for swing_kind, pool in ((SwingKind.HIGH, highs), (SwingKind.LOW, lows)):
        for cluster in _cluster(sorted(pool, key=lambda s: s.price), tolerance):
            zones.append(
                _build_zone(
                    df, cluster, swing_kind, resolved_atr, tolerance,
                    break_margin_atr, tick_size,
                )
            )

    zones = [z for z in zones if z.touches >= min_touches]
    zones.sort(key=lambda z: z.strength, reverse=True)
    return LevelMap(zones[:max_zones], resolved_atr)


def _cluster(by_price: List[SwingPoint], tolerance: float) -> List[List[SwingPoint]]:
    """
    Greedy 1-D clustering by price proximity.

    Sorted ascending, so a new swing joins the open cluster while it stays
    within ``tolerance`` of the running mean. The mean — not the first member —
    is the reference, so a long slow drift into a cluster does not walk the
    band upward one swing at a time.
    """
    if not by_price:
        return []

    clusters: List[List[SwingPoint]] = []
    current: List[SwingPoint] = [by_price[0]]

    for swing in by_price[1:]:
        mean = sum(s.price for s in current) / len(current)
        if abs(swing.price - mean) <= tolerance:
            current.append(swing)
        else:
            clusters.append(current)
            current = [swing]
    clusters.append(current)
    return clusters


def _build_zone(
    df: pd.DataFrame,
    cluster: List[SwingPoint],
    formed_as: SwingKind,
    atr: float,
    tolerance: float,
    break_margin_atr: float,
    tick_size: Optional[float],
) -> LevelZone:
    prices = [s.price for s in cluster]
    centre = sum(prices) / len(prices)
    raw_lower = min(prices)
    raw_upper = max(prices)

    first_index = min(s.index for s in cluster)
    last_index = max(s.index for s in cluster)

    # A level is a band, not a point. A cluster built from one swing has zero
    # intrinsic width, and a zero-width band can never be "reached" or
    # "contained" — reaction counting and proximity tests both need a real
    # zone. Pad by a quarter of the cluster tolerance: wide enough to mean
    # something, narrow enough that distinct levels cannot merge.
    pad = tolerance * 0.25
    lower = raw_lower - pad
    upper = raw_upper + pad

    if tick_size and tick_size > 0:
        centre = _snap(centre, tick_size)
        lower = _snap(lower, tick_size)
        upper = _snap(upper, tick_size)

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)

    # Touches: cluster members, plus every later bar that traded into the band.
    # Counting only the swings undervalues a level the market keeps revisiting
    # without producing a clean pivot each time.
    touches = len(cluster)
    for i in range(last_index + 1, n):
        if lows[i] <= upper and highs[i] >= lower:
            touches += 1

    broken, broken_index = _detect_break(
        df, lower, upper, formed_as, last_index, atr * break_margin_atr
    )

    kind = _effective_kind(formed_as, broken)
    strength, components = _score(
        touches=touches,
        last_index=last_index,
        n=n,
        broken=broken,
        broken_index=broken_index,
        raw_width=raw_upper - raw_lower,
        tolerance=tolerance,
        atr=atr,
    )

    return LevelZone(
        kind=kind,
        price=centre,
        lower=lower,
        upper=upper,
        formed_as=formed_as,
        touches=touches,
        strength=strength,
        first_index=first_index,
        last_index=last_index,
        broken=broken,
        broken_index=broken_index,
        components=components,
    )


def _detect_break(
    df: pd.DataFrame,
    lower: float,
    upper: float,
    formed_as: SwingKind,
    from_index: int,
    margin: float,
) -> tuple[bool, Optional[int]]:
    """
    Did a *close* decisively clear the zone after it formed?

    A wick through is not a break — it is liquidity being taken, and the two
    mean opposite things. The margin stops a graze from counting.
    """
    closes = df["close"].to_numpy(dtype=float)
    start = max(0, from_index + 1)

    if formed_as is SwingKind.HIGH:
        level = upper + margin
        for i in range(start, len(closes)):
            if closes[i] > level:
                return True, i
    else:
        level = lower - margin
        for i in range(start, len(closes)):
            if closes[i] < level:
                return True, i
    return False, None


def _effective_kind(formed_as: SwingKind, broken: bool) -> LevelKind:
    """
    A broken zone flips polarity.

    Support that price decisively closed below is not support any more; it is
    the level sellers defended on the way down, and the first thing a bounce
    tends to meet. This is why ``kind`` is computed rather than inherited.
    """
    if not broken:
        return LevelKind.RESISTANCE if formed_as is SwingKind.HIGH else LevelKind.SUPPORT
    return LevelKind.SUPPORT if formed_as is SwingKind.HIGH else LevelKind.RESISTANCE


def _score(
    touches: int,
    last_index: int,
    n: int,
    broken: bool,
    broken_index: Optional[int],
    raw_width: float,
    tolerance: float,
    atr: float,
) -> tuple[float, Dict[str, float]]:
    """
    Strength in 0..1 from four independent, defensible components.

    touch_score      reaction count, saturating — the 1st and 2nd touches tell
                     you far more than the 6th and 7th.
    recency_score    how recently the zone mattered, on a half-life over bars.
    freshness_score  *inverted* on recency-since-break: a zone nobody has come
                     back to since forming still holds resting orders and acts
                     as a magnet. A heavily retested zone is largely spent.
    tightness_score  how tightly the clustered swings agree, measured as the
                     cluster's real width against the tolerance that merged it.
                     A band the market repeatedly hit at the same price is a
                     better level than one that merely got grouped.

    A broken zone keeps a low ceiling rather than scoring zero: it often still
    matters as a flip level, and zero would throw that away.
    """
    touch_score = min(touches / 4.0, 1.0)
    age = max(0, (n - 1) - last_index)
    recency_score = float(np.exp(-age / 120.0))

    if broken:
        since = (n - 1) - (broken_index if broken_index is not None else last_index)
        freshness_score = float(np.exp(-since / 60.0)) * 0.5
    else:
        # Never revisited since the last swing: full strength.
        retests = max(0, touches - 1)
        freshness_score = float(np.exp(-retests / 3.0))

    if tolerance > 0:
        tightness_score = 1.0 - min(abs(raw_width) / tolerance, 1.0)
    else:
        tightness_score = 0.5

    weights = {
        "touch": 0.35,
        "recency": 0.25,
        "freshness": 0.30,
        "tightness": 0.10,
    }
    components = {
        "touch": touch_score,
        "recency": recency_score,
        "freshness": freshness_score,
        "tightness": tightness_score,
    }
    raw = sum(components[k] * w for k, w in weights.items())

    ceiling = 0.55 if broken else 1.0
    return min(raw, ceiling), components


def _snap(price: float, tick_size: float) -> float:
    """
    Snap to the broker's price grid, half-up.

    Half-up rather than Python's default so a level never lands a tick into the
    neighbouring zone. The final round kills the binary artefact that
    ``floor(x) * tick_size`` leaves behind (110.60000000000001 is not a price a
    broker will accept); the decimal count comes from ``tick_size`` itself, so
    no per-instrument digit count is assumed.
    """
    snapped = float(np.floor(price / tick_size + 0.5) * tick_size)
    for decimals in range(1, 11):
        rounded = round(snapped, decimals)
        if abs(rounded - snapped) <= tick_size * 1e-6:
            return rounded
    return snapped


    if not {"high", "low", "close"}.issubset(df.columns) or len(df) < 2:
        return None
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    last = float(tr.iloc[-1])
    return last if last > 0 else None