"""
core/market/location.py — Where is the market, right now?

This is the module that answers the question. Everything else in
``core/market/`` feeds it; everything downstream consumes the single
``MarketLocation`` it returns.

The central fact is **range position**. Once the active dealing range is known
— the span between the most recent confirmed swing high and swing low — every
price has an address inside it. At 80% of the range you are in *premium*:
you are paying more than the market has recently been willing to accept, and
the path of least resistance is down. At 20% you are in *discount*. This one
number is what the existing strategies were blind to, and it is why a long
entered at the top of a range scored identically to one entered at the bottom.

The gate below is deliberately a *veto*, not a scorer. ``core/strategy/`` has
three tested strategies producing EMA/RSI/MACD opinions; they remain the "what"
layer. This module only refuses locations where the "what" cannot possibly be
right — buying in premium against untaken sell-side liquidity, selling in
discount against untaken buy-side liquidity, or acting at all when there is no
range to be located within.

Honesty rules that are load-bearing here:

- No confirmed range means ``UNDEFINED``, and ``UNDEFINED`` **blocks**. The
  alternative — treating an unknown location as neutral and allowing the trade
  — is precisely the silent-default failure this codebase has already paid for
  once in the instrument registry.
- ``structural_stop`` returns ``None`` when it cannot place a defensible stop.
  ``None`` is not 0 and not "no stop": it means the caller must refuse.
- Every distance is reported in ATR alongside raw price, because a level
  "0.5 away" means nothing until you know the volatility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import pandas as pd

from core.market.levels import LevelMap, LevelZone, build_levels
from core.market.liquidity import LiquidityMap, LiquidityPool, LiquidityState, scan_liquidity
from core.market.structure import BreakKind, StructureBias, StructureState, active_swing, analyse_structure
from core.market.swings import SwingKind, SwingPoint, find_swings

from core.market.series import resolve_atr, valid_tick_size

logger = logging.getLogger(__name__)


class LocationBias(str, Enum):
    """Where price sits inside the active range."""

    PREMIUM = "PREMIUM"        # >= 80% — expensive; favour sellers
    UPPER_MID = "UPPER_MID"
    MID = "MID"
    LOWER_MID = "LOWER_MID"
    DISCOUNT = "DISCOUNT"      # <= 20% — cheap; favour buyers
    # No usable range. Never treated as "neutral".
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class StopPlan:
    """A stop derived from market structure rather than a blind ATR multiple."""

    price: float
    anchor: float              # the swing the stop protects
    reason: str
    distance: float            # absolute, in price
    distance_atr: float        # same distance in ATR
    structural: bool           # True when anchored to a swing

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "anchor": self.anchor,
            "reason": self.reason,
            "distance": self.distance,
            "distance_atr": round(self.distance_atr, 3),
            "structural": self.structural,
        }


@dataclass(frozen=True)
class EntryGate:
    """Verdict on whether a trade direction is sane at this location."""

    allowed: bool
    direction: int
    blocking: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        parts = list(self.blocking)
        parts.extend(self.cautions)
        return "; ".join(parts) if parts else "Location permits entry"

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "direction": self.direction,
            "blocking": list(self.blocking),
            "cautions": list(self.cautions),
        }


@dataclass(frozen=True)
class MarketLocation:
    """Complete spatial read of one symbol at one moment."""

    price: float
    atr: Optional[float]
    bias: LocationBias
    range_position: Optional[float]     # 0.0 = range low, 1.0 = range high
    range_high: Optional[float]
    range_low: Optional[float]
    structure: StructureState
    levels: LevelMap
    liquidity: LiquidityMap
    fib_levels: Dict[str, float] = field(default_factory=dict)
    swings: List[SwingPoint] = field(default_factory=list)
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)

    # ── convenience lookups ────────────────────────────────────────────────

    @property
    def has_range(self) -> bool:
        return self.range_high is not None and self.range_low is not None and self.range_high > self.range_low

    @property
    def below_range(self) -> bool:
        """Price has closed beneath the range low — a breakdown, not a discount."""
        return (
            self.has_range
            and self.price < self.range_low
        )

    @property
    def above_range(self) -> bool:
        """Price has closed above the range high — a breakout, not a premium."""
        return (
            self.has_range
            and self.price > self.range_high
        )

    @property
    def is_valid(self) -> bool:
        """Usable enough to trade against. ``UNDEFINED`` location is not."""
        return self.bias is not LocationBias.UNDEFINED

    def nearest_resistance(self) -> Optional[LevelZone]:
        return self.levels.nearest_above(self.price)

    def nearest_support(self) -> Optional[LevelZone]:
        return self.levels.nearest_below(self.price)

    def distance_to_resistance(self) -> Optional[float]:
        z = self.nearest_resistance()
        return None if z is None else z.lower - self.price

    def distance_to_support(self) -> Optional[float]:
        z = self.nearest_support()
        return None if z is None else self.price - z.upper

    def target_above(self) -> Optional[LiquidityPool]:
        """Untaken buy-side liquidity above — the natural profit objective."""
        for pool in self.liquidity.above(self.price):
            if pool.state is LiquidityState.RESTING:
                return pool
        return None

    def target_below(self) -> Optional[LiquidityPool]:
        for pool in self.liquidity.below(self.price):
            if pool.state is LiquidityState.RESTING:
                return pool
        return None

    # ── the two decisions downstream code actually needs ───────────────────

    def entry_gate(self, direction: int) -> EntryGate:
        """
        Veto a direction that location makes indefensible.

        Kept as a veto rather than a score modifier so that the existing
        strategy ensemble keeps its tested behaviour and its safety
        regressions, and location only ever *removes* trades.
        """
        if direction == 0:
            return EntryGate(False, 0, ["No direction requested"])

        side = "long" if direction > 0 else "short"
        blocking: List[str] = []
        cautions: List[str] = []

        if self.bias is LocationBias.UNDEFINED:
            blocking.append(
                "No confirmed range: location is undefined and cannot be traded against"
            )
            return EntryGate(False, direction, blocking)

        pos = self.range_position

        # Premium/discount is a *ranging* concept and must not be applied to a
        # trend as though it were one. In a confirmed uptrend price sits near,
        # or above, the top of its own swing range by construction — that is
        # what an uptrend looks like — so vetoing "premium" longs would block
        # essentially every trend trade. Worse, it inverts the meaning: price
        # above the range high *in the direction of the trend* is the strongest
        # location there is, not the weakest.
        #
        # The regime-aware rule:
        #   RANGING / UNDEFINED   strict: buy discount, sell premium.
        #   with the trend         permitted; only extension is cautioned.
        #   against the trend      only at the counter-extreme (a long in a bear
        #                          trend solely in DISCOUNT), never anywhere else.
        bias = self.structure.bias
        with_trend = (direction > 0 and bias is StructureBias.BULLISH) or (
            direction < 0 and bias is StructureBias.BEARISH
        )
        against_trend = (direction > 0 and bias is StructureBias.BEARISH) or (
            direction < 0 and bias is StructureBias.BULLISH
        )

        if against_trend:
            # Counter-trend is permitted only at the counter-extreme — buying a
            # bear trend solely in discount, selling a bull trend solely in
            # premium — and only while price is still *inside* the range.
            #
            # That second condition is the one that was missing. Without it a
            # confirmed uptrend running to new highs clamps to 100% of range,
            # reads as PREMIUM, and is waved through as a textbook "sell into
            # strength" — permitting a short at the exact moment price is
            # breaking out. The premium reading describes *where price is*; it
            # does not mean the trend has stopped.
            if direction < 0 and bias is StructureBias.BULLISH and self.above_range:
                blocking.append(
                    f"{side} rejected: price {self.price:.5f} has broken above the "
                    f"range high {self.range_high:.5f} in a confirmed BULLISH "
                    f"structure — counter-trend into a breakout"
                )
            if direction > 0 and bias is StructureBias.BEARISH and self.below_range:
                blocking.append(
                    f"{side} rejected: price {self.price:.5f} has broken below the "
                    f"range low {self.range_low:.5f} in a confirmed BEARISH "
                    f"structure — counter-trend into a breakdown"
                )

            if direction > 0 and self.bias is not LocationBias.DISCOUNT:
                blocking.append(
                    f"{side} rejected against a {bias.value} structure: "
                    f"counter-trend entries are only permitted in DISCOUNT, "
                    f"price is {self.bias.value} at {pos:.0%} of range"
                )
            if direction < 0 and self.bias is not LocationBias.PREMIUM:
                blocking.append(
                    f"{side} rejected against a {bias.value} structure: "
                    f"counter-trend entries are only permitted in PREMIUM, "
                    f"price is {self.bias.value} at {pos:.0%} of range"
                )
        elif not with_trend:
            # Ranging or structurally undefined: strict premium/discount.
            if direction > 0 and self.bias in (LocationBias.PREMIUM, LocationBias.UPPER_MID):
                blocking.append(
                    f"{side} rejected in {self.bias.value} "
                    f"(range position {pos:.0%}, structure {bias.value}): "
                    f"paying above the range midpoint"
                )
            if direction < 0 and self.bias in (LocationBias.DISCOUNT, LocationBias.LOWER_MID):
                blocking.append(
                    f"{side} rejected in {self.bias.value} "
                    f"(range position {pos:.0%}, structure {bias.value}): "
                    f"selling below the range midpoint"
                )
        else:
            # With the trend: location advises, and only extension is flagged.
            if direction > 0 and self.bias is LocationBias.PREMIUM and self.above_range:
                cautions.append(
                    f"Extended: price {self.price:.5f} is already above the range "
                    f"high {self.range_high:.5f} in PREMIUM — late entry into an "
                    f"advancing trend"
                )
            if direction < 0 and self.bias is LocationBias.DISCOUNT and self.below_range:
                cautions.append(
                    f"Extended: price {self.price:.5f} is already below the range "
                    f"low {self.range_low:.5f} in DISCOUNT — late entry into a "
                    f"declining trend"
                )

        # Entering directly into untaken opposing liquidity.
        pool = self.target_above() if direction > 0 else self.target_below()
        if pool is not None and self.atr and pool.members >= 2:
            gap = abs(pool.price - self.price)
            if 0 < gap <= self.atr * 0.75:
                cautions.append(
                    f"Untaken liquidity {pool.kind.value} pool at {pool.price:.5f} "
                    f"({pool.members} swings) only {gap / self.atr:.2f} ATR away"
                )

        # Price has left the dealing range entirely. This is a break, not an
        # extreme in the range, and the premium/discount reasoning above does
        # not describe it — so it is surfaced rather than absorbed.
        if self.below_range:
            cautions.append(
                f"Price {self.price:.5f} is below the range low {self.range_low:.5f}: "
                f"range breakdown, not a discount entry"
            )
        elif self.above_range:
            cautions.append(
                f"Price {self.price:.5f} is above the range high {self.range_high:.5f}: "
                f"range breakout, not a premium entry"
            )

        # Trading straight into a strong level on the far side of the move.
        far = (
            self.distance_to_resistance() if direction > 0 else self.distance_to_support()
        )
        if far is not None and self.atr and far <= self.atr * 0.5:
            level = self.nearest_resistance() if direction > 0 else self.nearest_support()
            if level is not None and level.strength >= 0.5:
                cautions.append(
                    f"Strong {level.kind.value.lower()} at {level.price:.5f} "
                    f"({far / self.atr:.2f} ATR away, strength {level.strength:.2f})"
                )

        # A fresh CHoCH against the trade is the strongest warning available.
        last = self.structure.last_choch
        if last is not None and last.direction != direction:
            cautions.append(
                f"Last structural break was a {last.kind.value} against this "
                f"{side} ({last.level:.5f})"
            )

        return EntryGate(not blocking, direction, blocking, cautions)

    def structural_stop(
        self,
        entry: float,
        direction: int,
        min_atr_mult: float = 0.75,
        max_atr_mult: float = 3.0,
        tick_size: Optional[float] = None,
    ) -> Optional[StopPlan]:
        """
        Place the stop where the market says the idea is wrong.

        For a long that is just below the swing low the buyers are defending;
        for a short, just above the swing high. This answers with a *price* the
        question ``Signal.invalidation`` can only answer in English.

        The ATR bounds are safety rails in both directions: the floor keeps the
        stop out of normal bar noise (where it would be taken out and the thesis
        would be right), the ceiling stops a swing so distant that the resulting
        position is un-sizably small. Neither bound silently invents a level —
        when there is no swing to work from, the answer is ``None``.

        Returns:
            A ``StopPlan``, or ``None`` when no defensible stop exists.
        """
        if direction == 0:
            return None

        tick_size = valid_tick_size(tick_size)
        atr = self.atr
        if not atr or atr <= 0:
            return None

        anchor = active_swing(self.structure, direction)
        structural = False
        reason = ""
        stop: Optional[float] = None

        if anchor is not None:
            if direction > 0 and anchor.price < entry:
                stop = anchor.price
                structural = True
                reason = f"Below swing low {anchor.price:.5f} ({anchor.time})"
            elif direction < 0 and anchor.price > entry:
                stop = anchor.price
                structural = True
                reason = f"Above swing high {anchor.price:.5f} ({anchor.time})"

        if stop is None:
            # No usable swing in the trade's favour. Fall back to a volatility
            # stop, which is honest about being non-structural.
            stop = entry - direction * atr * min_atr_mult
            reason = f"No swing available; {min_atr_mult:.2f} ATR volatility stop"

        # Apply bounds, keeping the stop on the correct side of entry.
        distance = abs(entry - stop)
        floor = atr * min_atr_mult
        ceiling = atr * max_atr_mult

        if distance < floor:
            stop = entry - direction * floor
            reason = f"Volatility floor {min_atr_mult:.2f} ATR (structure too tight)"
            structural = False
        elif distance > ceiling:
            stop = entry - direction * ceiling
            reason = f"Volatility ceiling {max_atr_mult:.2f} ATR (structure too distant)"
            structural = False

        if tick_size and tick_size > 0:
            from core.market.levels import _snap

            stop = _snap(stop, tick_size)
            # Snapping is direction-blind, so it can pull the stop back inside
            # the floor or push it past the ceiling. Both bounds are promises
            # about the final number, so re-assert them after rounding: step
            # outward to honour the floor, inward to honour the ceiling.
            snapped_distance = abs(entry - stop)
            if snapped_distance < floor:
                stop -= direction * tick_size * (
                    int((floor - snapped_distance) // tick_size) + 1
                )
            elif snapped_distance > ceiling:
                stop += direction * tick_size * (
                    int((snapped_distance - ceiling) // tick_size) + 1
                )

        final_distance = abs(entry - stop)

        # Refuse a degenerate stop outright. A near-zero stop distance is not a
        # conservative stop, it is a dangerous one: ``calculate_lot_size``
        # divides by stop distance, so a collapsed stop sizes a maximal
        # position that is then exposed to being stopped by one tick of noise.
        # There is no defensible stop here, so there is no stop.
        if final_distance <= 0 or final_distance <= abs(entry) * 1e-9:
            logger.warning(
                "structural_stop: refusing degenerate stop "
                "(entry=%r atr=%r distance=%r)",
                entry, atr, final_distance,
            )
            return None

        return StopPlan(
            price=stop,
            anchor=anchor.price if anchor else entry,
            reason=reason,
            distance=final_distance,
            distance_atr=final_distance / atr,
            structural=structural,
        )

    def describe(self) -> str:
        """One-line human summary, suitable for logs and the dashboard."""
        if self.bias is LocationBias.UNDEFINED:
            return "Location UNDEFINED (no confirmed range)"
        where = f"{self.bias.value} @ {self.range_position:.0%} of range"
        if self.below_range:
            where = f"BELOW RANGE ({self.bias.value})"
        elif self.above_range:
            where = f"ABOVE RANGE ({self.bias.value})"
        bits = [
            where,
            f"{self.range_low:.5f}-{self.range_high:.5f}",
            f"structure {self.structure.bias.value}",
        ]
        if self.atr:
            bits.append(f"ATR {self.atr:.5f}")
        res = self.nearest_resistance()
        if res and self.atr:
            bits.append(f"res {res.price:.5f} ({self.distance_to_resistance() / self.atr:.2f} ATR)")
        sup = self.nearest_support()
        if sup and self.atr:
            bits.append(f"sup {sup.price:.5f} ({self.distance_to_support() / self.atr:.2f} ATR)")
        return " | ".join(bits)

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "atr": self.atr,
            "bias": self.bias.value,
            "range_position": self.range_position,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "fib_levels": dict(self.fib_levels),
            "structure": self.structure.to_dict(),
            "levels": self.levels.to_dict(),
            "liquidity": self.liquidity.to_dict(),
        }


def analyse_location(
    df: pd.DataFrame,
    swing_left: int = 2,
    swing_right: int = 2,
    atr: Optional[float] = None,
    tick_size: Optional[float] = None,
    premium_threshold: float = 0.80,
    discount_threshold: float = 0.20,
) -> MarketLocation:
    """
    Full location read for one symbol.

    Args:
        df: OHLCV frame, ideally with indicators from ``market.pipeline``.
        swing_left/right: Fractal window for pivot detection.
        atr: Current ATR; derived from the frame when absent.
        tick_size: Broker increment for snapping levels and stops.
        premium_threshold: Range position at or above which price is PREMIUM.
        discount_threshold: Range position at or below which price is DISCOUNT.

    Returns:
        A ``MarketLocation``. Never raises on bad input — insufficient data
        yields ``bias=UNDEFINED``, which callers must treat as "do not trade".
    """
    empty_structure = StructureState(StructureBias.UNDEFINED, 0.0, reasons=["No data"])

    if df is None or df.empty or "close" not in df.columns:
        return MarketLocation(
            price=0.0, atr=None, bias=LocationBias.UNDEFINED, range_position=None,
            range_high=None, range_low=None, structure=empty_structure,
            levels=LevelMap(), liquidity=LiquidityMap(), reasons=["No data"],
        )

    price = float(df["close"].iloc[-1])
    resolved_atr = atr if (atr is not None and atr > 0) else resolve_atr(df)

    swings_all = find_swings(df, left=swing_left, right=swing_right, include_unconfirmed=True)
    swings_confirmed = [s for s in swings_all if s.confirmed]

    structure = analyse_structure(df, left=swing_left, right=swing_right)
    levels = build_levels(
        df, swings_all, atr=resolved_atr, tick_size=tick_size,
        tolerance_atr=0.30,
    )
    liquidity = scan_liquidity(
        df, swings_all, atr=resolved_atr, tick_size=tick_size,
        tolerance_atr=0.20,
    )

    range_high, range_low = _active_range(swings_confirmed, resolved_atr)
    bias, position, fib = _range_position(
        price, range_high, range_low, premium_threshold, discount_threshold
    )

    reasons: List[str] = list(structure.reasons)
    if bias is LocationBias.UNDEFINED:
        reasons.append("No usable dealing range: refusing to locate price")
    else:
        reasons.append(
            f"Range {range_low:.5f}-{range_high:.5f}; price at {position:.1%} "
            f"({bias.value})"
        )

    confidence = structure.confidence if bias is not LocationBias.UNDEFINED else 0.0

    return MarketLocation(
        price=price,
        atr=resolved_atr,
        bias=bias,
        range_position=position,
        range_high=range_high,
        range_low=range_low,
        structure=structure,
        levels=levels,
        liquidity=liquidity,
        fib_levels=fib,
        swings=swings_all,
        confidence=confidence,
        reasons=reasons,
    )


def _active_range(
    swings: Sequence[SwingPoint], atr: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """
    The active dealing range: the most recent confirmed swing of each kind.

    Returns ``(None, None)`` when either side is missing or the two are too
    close to constitute a range. A 2-pip span is not a range to trade the edges
    of, and reporting it as one would put "premium" and "discount" within a
    hair of each other.
    """
    highs = [s for s in swings if s.kind is SwingKind.HIGH]
    lows = [s for s in swings if s.kind is SwingKind.LOW]
    if not highs or not lows:
        return None, None

    high = highs[-1].price
    low = lows[-1].price
    if high <= low:
        return None, None

    if atr and (high - low) < atr * 1.5:
        # Too tight to be a range.
        return None, None

    return high, low


def _range_position(
    price: float,
    high: Optional[float],
    low: Optional[float],
    premium_threshold: float,
    discount_threshold: float,
) -> tuple[LocationBias, Optional[float], Dict[str, float]]:
    """Classify price inside the range and compute Fibonacci retracements."""
    if high is None or low is None or high <= low:
        return LocationBias.UNDEFINED, None, {}

    span = high - low
    raw_position = (price - low) / span

    # Clamp into [0, 1]. A price below the range low is not an infinitely deep
    # discount — it has left the range entirely, which is a different fact and
    # is reported separately via ``below_range`` / ``above_range``. Silently
    # scoring a breakdown as "very cheap" would make the most violent move in
    # the data look like the most attractive entry.
    position = min(max(raw_position, 0.0), 1.0)

    fib = {
        "0.0": low,
        "0.236": low + span * 0.236,
        "0.382": low + span * 0.382,
        "0.5": low + span * 0.5,
        "0.618": low + span * 0.618,
        "0.705": low + span * 0.705,
        "0.786": low + span * 0.786,
        "1.0": high,
    }

    if position >= premium_threshold:
        bias = LocationBias.PREMIUM
    elif position >= 0.60:
        bias = LocationBias.UPPER_MID
    elif position >= 0.40:
        bias = LocationBias.MID
    elif position > discount_threshold:
        bias = LocationBias.LOWER_MID
    else:
        bias = LocationBias.DISCOUNT

    return bias, position, fib


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