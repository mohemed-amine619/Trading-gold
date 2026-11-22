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


def withdrawal_peak_reduction(
    peak_equity: Optional[float],
    equity: Optional[float],
    withdrawn: Optional[float],
) -> float:
    """How much to LOWER the peak-to-trough reference after a withdrawal.

    The total-drawdown ceiling is measured as ``(peak - equity) / peak``. Cash
    taken out of the account lowers equity without a trade being lost, so the
    peak has to follow it down. Otherwise any sizeable withdrawal halts the bot
    permanently: taking $1000 out of a $1500 account reads as a 66% drawdown
    against a 20% cap, and only a manual reset clears it.

    The caller must establish ``withdrawn`` from the broker's **balance
    operations** (``DEAL_TYPE_BALANCE``), never by inferring it from a fall in
    equity. A losing trade and a withdrawal both lower equity and cannot be
    told apart from equity alone — and lowering the peak for a *loss* would
    erase the very drawdown this ceiling exists to catch. That ambiguity is
    why ``deposit_baseline_adjustment`` refuses to act in the other direction;
    here the broker states which it was, so the ambiguity does not arise.

    Returns the positive amount to subtract from ``peak_equity``, bounded so
    the peak can never drop below ``equity`` (that would invent a profit) and
    never below zero. Unknown or non-positive inputs return 0.0 (fail closed).
    Never raises.
    """
    peak = finite_or_none(peak_equity)
    eq = finite_or_none(equity)
    out = finite_or_none(withdrawn)
    if peak is None or eq is None or out is None:
        return 0.0
    if peak <= 0 or out <= 0 or eq < 0:
        return 0.0

    # Already at or below the peak reference: no drawdown to relieve.
    headroom = float(peak) - float(eq)
    if headroom <= 0:
        return 0.0

    # Lower by the cash removed, but never past current equity.
    return min(float(out), headroom)
