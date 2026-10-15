"""
core/market/liquidity.py — Resting liquidity and stop sweeps.

Retail stops cluster in predictable places: just above obvious swing highs and
just below obvious swing lows. That makes those prices not just levels but
*targets* — the market is drawn to them because that is where the orders sit.

The distinction that matters is not the pool, it is what happened to it:

- **RESTING** — an untaken cluster. Often above the current price, acting as a
  magnet. This is a legitimate profit target, and the reason "where is the next
  pool" is a real question rather than a decorative one.
- **SWEPT** — price traded through it (wick) and the bar closed back inside.
  Stops were filled and the level was rejected on the same bar. In price
  action this is one of the higher-quality reversal facts available, and it is
  the direct counterpoint to the body-close rule used in
  ``core/market/structure.py``: a wick through a level is liquidity being
  taken, *not* a structural break, and treating the two as the same signal is
  how a system ends up calling a violent reversal a bullish breakout.
- **MITIGATED** — swept and then rejected decisively. The pool is spent.

This is deliberately separate from ``core/market/levels.py``. Levels answer
"where did price turn"; liquidity answers "where are the stops".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from core.market.swings import SwingKind, SwingPoint

from core.market.series import resolve_atr, valid_tick_size

logger = logging.getLogger(__name__)


class LiquidityState(str, Enum):
    RESTING = "RESTING"
    SWEPT = "SWEPT"
    MITIGATED = "MITIGATED"


@dataclass(frozen=True)
class LiquidityPool:
    """
    A cluster of stop orders at approximately one price.

    Attributes:
        kind: ``HIGH`` for buy-side liquidity (stops above swing highs),
            ``LOW`` for sell-side liquidity (stops below swing lows).
        price: Pool level.
        members: How many swings share this price. Two or more is what makes
            it a *pool* rather than an ordinary level.
        state: RESTING / SWEPT / MITIGATED.
        swept_index, swept_time: First bar that took the pool.
        rejection_atr: Distance price moved away from the pool after the sweep,
            in ATR. Measures whether the sweep was actually rejected or the
            market just kept going.
    """

    kind: SwingKind
    price: float
    members: int = 1
    state: LiquidityState = LiquidityState.RESTING
    swept_index: Optional[int] = None
    swept_time: Optional[pd.Timestamp] = None
    rejection_atr: float = 0.0
    first_index: int = 0
    last_index: int = 0

    @property
    def is_above(self) -> bool:
        """Buy-side liquidity — sits above price, a magnet/target."""
        return self.kind is SwingKind.HIGH

    @property
    def is_below(self) -> bool:
        """Sell-side liquidity — sits below price, a magnet/target."""
        return self.kind is SwingKind.LOW

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "price": self.price,
            "members": self.members,
            "state": self.state.value,
            "swept_index": self.swept_index,
            "swept_time": self.swept_time,
            "rejection_atr": round(self.rejection_atr, 3),
            "first_index": self.first_index,
            "last_index": self.last_index,
        }


@dataclass(frozen=True)
class LiquidityMap:
    pools: List[LiquidityPool] = field(default_factory=list)
    atr: Optional[float] = None

    def above(self, price: float) -> List[LiquidityPool]:
        """Buy-side pools above ``price``, nearest first."""
        out = [p for p in self.pools if p.is_above and p.price > price]
        return sorted(out, key=lambda p: p.price)

    def below(self, price: float) -> List[LiquidityPool]:
        """Sell-side pools below ``price``, nearest first."""
        out = [p for p in self.pools if p.is_below and p.price < price]
        return sorted(out, key=lambda p: p.price, reverse=True)

    def nearest_target_above(self, price: float) -> Optional[LiquidityPool]:
        got = self.above(price)
        return got[0] if got else None

    def nearest_target_below(self, price: float) -> Optional[LiquidityPool]:
        got = self.below(price)
        return got[0] if got else None

    def swept_below(self, price: float) -> List[LiquidityPool]:
        """Pools below price that were taken — recent reversal evidence."""
        return [p for p in self.below(price) if p.state is not LiquidityState.RESTING]

    def swept_above(self, price: float) -> List[LiquidityPool]:
        return [p for p in self.above(price) if p.state is not LiquidityState.RESTING]

    def to_dict(self) -> dict:
        return {"atr": self.atr, "pools": [p.to_dict() for p in self.pools]}


def scan_liquidity(
    df: pd.DataFrame,
    swings: Sequence[SwingPoint],
    atr: Optional[float] = None,
    tolerance_atr: float = 0.20,
    min_members: int = 2,
    rejection_atr: float = 0.75,
    max_pools: int = 16,
    tick_size: Optional[float] = None,
) -> LiquidityMap:
    """
    Find stop pools and whether they are resting, swept, or mitigated.

    Args:
        df: OHLCV frame.
        swings: Swings to cluster. Unconfirmed ones are ignored.
        atr: Current ATR; derived from true range when absent.
        tolerance_atr: Two swings within this multiple of ATR share a pool.
            Deliberately tighter than the level tolerance in ``levels.py``:
            equal highs must be *genuinely* equal for stops to be stacked
            there, and a loose tolerance would report every level as a pool.
        min_members: Swings required to qualify as a pool.
        rejection_atr: How far price must retreat after a sweep before the pool
            counts as MITIGATED rather than merely SWEPT.
        max_pools: Bound on returned pools.
        tick_size: Broker increment for snapping pool prices.

    Returns:
        A ``LiquidityMap``. Empty when no scale or no multi-member clusters.
    """
    if df is None or df.empty:
        return LiquidityMap([], atr)

    confirmed = [s for s in swings if s.confirmed]
    if len(confirmed) < min_members:
        return LiquidityMap([], atr)

    tick_size = valid_tick_size(tick_size)
    resolved_atr = atr if (atr is not None and atr > 0) else resolve_atr(df)
    if resolved_atr is None or resolved_atr <= 0:
        logger.debug("scan_liquidity: no usable ATR, refusing to cluster")
        return LiquidityMap([], None)

    tolerance = resolved_atr * tolerance_atr
    if tick_size and tick_size > 0:
        tolerance = max(tolerance, tick_size)

    pools: List[LiquidityPool] = []

    for kind in (SwingKind.HIGH, SwingKind.LOW):
        members = sorted(
            (s for s in confirmed if s.kind is kind), key=lambda s: s.price
        )
        for cluster in _cluster(members, tolerance):
            if len(cluster) < min_members:
                continue
            price = sum(s.price for s in cluster) / len(cluster)
            if tick_size and tick_size > 0:
                price = _snap(price, tick_size)
            pools.append(
                _build_pool(df, cluster, kind, price, resolved_atr, rejection_atr)
            )

    pools.sort(key=lambda p: (p.state is not LiquidityState.RESTING, p.price))
    return LiquidityMap(pools[:max_pools], resolved_atr)


def _cluster(by_price: List[SwingPoint], tolerance: float) -> List[List[SwingPoint]]:
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


def _build_pool(
    df: pd.DataFrame,
    cluster: List[SwingPoint],
    kind: SwingKind,
    price: float,
    atr: float,
    rejection_atr: float,
) -> LiquidityPool:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times = df.index

    first_index = min(s.index for s in cluster)
    last_index = max(s.index for s in cluster)

    # A sweep is a wick beyond the pool that closes back inside it. Requiring
    # the close back inside is what separates "stops were filled and rejected"
    # from "price broke out cleanly", which is a different event entirely.
    #
    # Scanning starts at ``first_index + 1``, NOT ``last_index + 1``. A sweep
    # bar forms its own swing high, so that swing lands inside the cluster —
    # starting after the cluster would skip the very bar that took the stops,
    # and every sweep would be reported as RESTING forever.
    # A sweep must close back beyond the zone, not merely inside it. Without
    # this margin every retest of a level registers as a stop run.
    sweep_margin = max(atr * 0.15, 1e-9)
    swept_index = _find_sweep(highs, lows, closes, first_index, price, kind, sweep_margin)

    if swept_index is None:
        return LiquidityPool(
            kind=kind,
            price=price,
            members=len(cluster),
            state=LiquidityState.RESTING,
            first_index=first_index,
            last_index=last_index,
        )

    # The wick that took the stops is not part of the level being defended.
    # Re-base the pool on the members that formed *before* the sweep so the
    # reported price is the resting level, not a level dragged upward by the
    # stop-runner. Fall back to the full cluster when every member came later.
    pre_sweep = [s for s in cluster if s.index < swept_index]
    level = (
        sum(s.price for s in pre_sweep) / len(pre_sweep) if pre_sweep else price
    )

    if kind is SwingKind.HIGH:
        worst = float(np.min(closes[swept_index:])) if len(closes) > swept_index else level
        rejection = (level - worst) / atr
    else:
        best = float(np.max(closes[swept_index:])) if len(closes) > swept_index else level
        rejection = (best - level) / atr

    state = (
        LiquidityState.MITIGATED if rejection >= rejection_atr else LiquidityState.SWEPT
    )

    return LiquidityPool(
        kind=kind,
        price=level,
        members=len(cluster),
        state=state,
        swept_index=swept_index,
        swept_time=times[swept_index],
        rejection_atr=max(rejection, 0.0),
        first_index=first_index,
        last_index=last_index,
    )


def _find_sweep(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    start: int,
    price: float,
    kind: SwingKind,
    margin: float,
) -> Optional[int]:
    """
    First bar after ``start`` that wicks through ``price`` and closes back
    decisively on the far side.

    ``margin`` is essential. Testing only ``close < price`` classifies a bar
    that pokes 0.1 above the pool and closes 0.1 under it as a stop run — but
    that is just noise trading inside the zone, and it will fire on almost
    every retest. A sweep has to close back *beyond the zone*, by at least its
    own width, before it counts as stops having been filled.
    """
    for i in range(max(0, start), len(closes)):
        if kind is SwingKind.HIGH:
            if highs[i] > price and closes[i] < price - margin:
                return i
        else:
            if lows[i] < price and closes[i] > price + margin:
                return i
    return None


def _snap(price: float, tick_size: float) -> float:
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