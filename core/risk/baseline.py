"""
core/risk/baseline.py — keep external cash out of the daily risk baseline.

The daily drawdown cap and the daily profit target are both measured from
``daily_start``.  A deposit raises equity without a trade, so a naive
``equity - daily_start`` reads the deposit as profit: on a live account this
can latch the daily profit target and stop every entry for the rest of the day.

This module is deliberately conservative.  It only reports an adjustment when
the account is flat and no trade has been made today.  In that state equity can
only have risen through an external cash inflow.  As soon as there is a trade
or an open position, profit and a deposit are indistinguishable from the data
available to the bot, and the baseline must be left alone — rebase risk state
would silently erase a real drawdown or a real win.

Pure functions, no broker or I/O dependency.  Safe to import on any platform.
"""

from __future__ import annotations

from typing import Optional

from core.risk.daily_state import finite_or_none


def deposit_baseline_adjustment(
    daily_start: Optional[float],
    equity: Optional[float],
    balance: Optional[float],
    trades_today: Optional[int],
    open_positions: Optional[int],
) -> float:
    """External cash inflow to add to ``daily_start``, or 0.0.

    Returns a positive amount only when *all* of the following hold:

    * ``daily_start``, ``equity`` and ``balance`` are finite and the baseline
      is positive (an unknown baseline must never be rebased);
    * ``trades_today == 0`` and ``open_positions == 0`` — no trading activity
      that could explain the rise in equity;
    * both equity and balance are above the baseline.  The smaller of the two
      is used so an inconsistent broker snapshot can only understate, never
      overstate, the adjustment.

    A withdrawal (negative delta) is never rebased: lowering the baseline
    would re-arm the daily loss budget and hide a drawdown.  Unknown inputs
    return 0.0 (fail closed).  Never raises.
    """
    start = finite_or_none(daily_start)
    eq = finite_or_none(equity)
    bal = finite_or_none(balance)
    if start is None or eq is None or bal is None or start <= 0:
        return 0.0

    trades = finite_or_none(trades_today)
    positions = finite_or_none(open_positions)
    if trades is None or positions is None:
        return 0.0
    if trades != 0 or positions != 0:
        return 0.0

    delta = min(float(eq), float(bal)) - float(start)
    return delta if delta > 0 else 0.0
