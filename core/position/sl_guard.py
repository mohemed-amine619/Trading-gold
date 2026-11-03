"""
core/position/sl_guard.py — the two rules every stop change must pass.

Pure functions, no broker. They exist as a separate module because the "never
widen a stop" rule is the most expensive one to get wrong and the easiest to
test in isolation.

* ``is_tighter``: a stop may only move toward price. For a buy the new stop must
  be strictly higher than the current one, for a sell strictly lower. A current
  stop of 0 (or None) means "no stop set", so any stop counts as tighter.
* ``step_ok``: a stop change must be worth a broker round trip. It must improve
  the stop by at least ``min_step`` price units, EXCEPT a move that first
  reaches break-even, which is a change of state (the trade can no longer lose)
  and must land even when it is small.
"""

from __future__ import annotations

from typing import Optional

_EPS = 1e-9


def is_tighter(is_buy: bool, current_sl: Optional[float], new_sl: Optional[float]) -> bool:
    """True if ``new_sl`` is strictly tighter than ``current_sl``."""
    if new_sl is None or new_sl <= 0:
        return False
    current = float(current_sl or 0.0)
    if current <= 0:
        return True
    return new_sl > current + _EPS if is_buy else new_sl < current - _EPS


def step_ok(is_buy: bool, current_sl: Optional[float], new_sl: Optional[float],
            min_step: float, breakeven_sl: Optional[float] = None) -> bool:
    """True if the change is tighter AND large enough, or it reaches break-even.

    ``breakeven_sl`` is the break-even stop for the trade (entry plus buffer).
    A move that takes the stop from the wrong side of it to at-or-past it
    bypasses ``min_step``.
    """
    if not is_tighter(is_buy, current_sl, new_sl):
        return False
    current = float(current_sl or 0.0)
    if current <= 0:
        return True
    if breakeven_sl is not None:
        before_be = current < breakeven_sl - _EPS if is_buy else current > breakeven_sl + _EPS
        reaches_be = new_sl >= breakeven_sl - _EPS if is_buy else new_sl <= breakeven_sl + _EPS
        if before_be and reaches_be:
            return True
    improvement = (new_sl - current) if is_buy else (current - new_sl)
    return improvement >= min_step - _EPS
