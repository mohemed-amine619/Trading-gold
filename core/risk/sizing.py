"""
core/risk/sizing.py — position size that cannot silently exceed the risk budget.

The legacy sizer returned the broker's minimum lot whenever the risk per lot
was unknown, and floored every result to the minimum lot even when that one
lot risked several times the per-trade budget. On a small account that makes
the configured risk fraction fiction: the minimum lot, not the budget,
decides the real risk.

This module keeps the legacy volume formula exactly and adds two things:

* unusable broker data (missing or non-positive tick size / value / step,
  stop distance or equity) yields ``None``: "cannot size". It is never
  treated as the minimum lot.
* the money actually at risk is reported, and an optional ``max_overshoot``
  refuses an entry whose forced minimum lot risks more than that multiple of
  the budget. ``max_overshoot <= 0`` disables the refusal and keeps the legacy
  behaviour; the overshoot is still reported so the caller can log it.

Pure function, no broker import, no state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SizeResult:
    """Outcome of one sizing request.

    ``volume`` is ``None`` when the entry must be refused; ``reason`` then says
    why. ``risk_amount`` is the money lost if the stop is hit at ``volume``.
    ``forced_min_lot`` is True when the budget bought less than the minimum lot
    and the minimum lot was used anyway.
    """

    volume: Optional[float]
    risk_amount: Optional[float]
    budget: float
    forced_min_lot: bool = False
    reason: str = ""


def _positive(value) -> bool:
    return value is not None and value > 0


def size_within_budget(
    risk_budget: float,
    sl_distance: float,
    tick_size: Optional[float],
    tick_value: Optional[float],
    vol_min: Optional[float],
    vol_max: Optional[float],
    vol_step: Optional[float],
    max_overshoot: float = 0.0,
) -> SizeResult:
    """Size a position so that a stop-out risks about ``risk_budget`` money.

    ``risk_budget`` is the money the caller is willing to lose on this trade
    (equity x risk fraction x every scale factor). The volume is the budget
    divided by the loss per lot, floored to ``vol_step`` and clamped to
    ``[vol_min, vol_max]``.
    """
    budget = float(risk_budget or 0.0)

    for name, value in (("sl_distance", sl_distance), ("tick_size", tick_size),
                        ("tick_value", tick_value), ("vol_min", vol_min),
                        ("vol_max", vol_max), ("vol_step", vol_step)):
        if not _positive(value):
            return SizeResult(None, None, budget,
                              reason=f"cannot size: {name} missing or not positive")
    if budget <= 0:
        return SizeResult(None, None, budget,
                          reason="cannot size: risk budget is not positive")
    if vol_max < vol_min:
        return SizeResult(None, None, budget,
                          reason="cannot size: vol_max below vol_min")

    # Same expression as the legacy sizer, so volumes do not move.
    loss_per_lot = sl_distance / (tick_size + 1e-10) * tick_value
    raw = budget / loss_per_lot
    volume = math.floor(raw / vol_step) * vol_step
    volume = min(volume, vol_max)

    forced = volume < vol_min
    if forced:
        volume = vol_min

    risk_amount = volume * loss_per_lot

    if forced and max_overshoot > 0 and risk_amount > budget * max_overshoot:
        return SizeResult(
            None, risk_amount, budget, forced_min_lot=True,
            reason=(f"minimum lot {vol_min} would risk {risk_amount:.2f}, "
                    f"{risk_amount / budget:.1f}x the {budget:.2f} budget "
                    f"(limit {max_overshoot:.1f}x)"),
        )

    return SizeResult(volume, risk_amount, budget, forced_min_lot=forced)
