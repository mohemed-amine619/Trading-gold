"""
core/position/scratch.py — close a trade that never went the predicted way.

Pure functions, no broker. A separate module for the same reason as
``sl_guard.py``: this one decides to close a live position, so it has to be
trivially testable in isolation.

Why it exists (diagnostics, 2026-11-23, 23 trades):

    | peak profit reached | trades | win rate | avg net |
    |---------------------|--------|----------|---------|
    | <0.3 ATR            |   12   |   0.00   |  -2.123 |
    | 0.3-0.6             |    1   |   0.00   |  -2.400 |
    | 0.6-1.0             |    4   |   1.00   |  +0.297 |
    | >=1.0 ATR           |    5   |   1.00   |  +1.012 |

Every trade that reached 0.6 ATR won. Every trade that did not, lost. Of 13
losers, exactly one ever showed +0.3 ATR and none showed +1.0. Those 12
never-got-going trades each rode to the full ~1.3 ATR stop at -2.12.

The rule: once a trade has had long enough to show something and still has
not, stop paying the full stop distance and take the small loss.

**Judged on PEAK profit, never on current profit.** That distinction is the
whole safety of this module. A trade that ran to +0.8 ATR and came back has
not "failed to go the predicted way" -- it is a giveback, and
``profit_giveback_pct`` plus the trailing/lock logic already own that case.
Scratching on current profit would cut winners during normal pullbacks, and
cutting winners early is already the single largest measured problem in this
account (winners kept a median 25% of their peak).

This is NOT a profit rule and does not create an edge. It converts a known
distribution of large losses into smaller ones. On the sample above, cutting
the average loss from 2.14 to about 1.0 moves expectancy from -0.94 to about
-0.29 per trade -- still negative. Validate the real size of that improvement
with ``scripts/replay_management.py`` against actual bars before trusting it.

Deliberately conservative:

* OFF unless ``after_s`` is positive. An unset configuration changes nothing.
* A trade whose peak cleared the bar is never scratched, whatever the clock.
* Unknown or non-finite inputs refuse (fail closed). Never raises.
"""

from __future__ import annotations

from typing import Optional

from core.risk.daily_state import finite_or_none


def should_scratch(
    peak_profit: Optional[float],
    atr: Optional[float],
    elapsed_s: Optional[float],
    after_s: Optional[float],
    min_peak_atr: Optional[float],
) -> bool:
    """True when a position has run out of time without showing enough profit.

    ``peak_profit`` is the best favourable excursion in PRICE UNITS since entry
    (``_peak_profit[ticket]`` in run.py), ``atr`` is in the same units, and
    ``elapsed_s`` is seconds since the position was first seen.

    ``after_s <= 0`` disables the rule entirely, which is the default, so an
    untouched configuration behaves exactly as it did before.

    Returns False for any input that cannot be read as a finite number: an exit
    that cannot be justified must not fire.
    """
    budget = finite_or_none(after_s)
    if budget is None or budget <= 0:
        return False                      # disabled

    elapsed = finite_or_none(elapsed_s)
    if elapsed is None or elapsed < budget:
        return False                      # still inside its time budget

    a = finite_or_none(atr)
    if a is None or a <= 0:
        return False                      # no scale to judge "enough" against

    threshold = finite_or_none(min_peak_atr)
    if threshold is None or threshold <= 0:
        return False                      # no bar to clear => nothing can fail it

    peak = finite_or_none(peak_profit)
    if peak is None:
        return False
    # A negative peak means the trade was never in profit at all. That is
    # precisely the case this rule targets, so treat it as 0 rather than refuse.
    peak = max(0.0, float(peak))

    return peak < threshold * float(a)
