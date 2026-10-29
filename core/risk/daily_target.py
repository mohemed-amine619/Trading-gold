"""
core/risk/daily_target.py — daily profit lock (gain protection).

Once the day's equity has risen far enough above the day's starting equity,
the caller stops opening new positions for the rest of the day. Open
positions are not touched: the lock only removes new risk, it never closes
a winner early or widens a stop.

This is the mirror image of the daily drawdown cap. The cap stops the bot
from giving back too much; the lock stops it from giving back a good day.
It does not make the bot win, and it does not raise size or trade count to
reach a target.

Pure functions, no state. The caller supplies the day's starting equity and
the current equity. Equity includes floating P&L, so the lock is measured
on the same basis as the drawdown cap.
"""

from __future__ import annotations


def daily_gain(daily_start: float, equity: float) -> float:
    """Fractional gain on the day's starting equity (0.05 = +5%).

    Returns 0.0 when the starting equity is unknown or not positive. A
    missing baseline must never read as a profit, so it never trips the lock.
    """
    if daily_start is None or daily_start <= 0:
        return 0.0
    return (equity - daily_start) / daily_start


def profit_lock_reached(daily_start: float, equity: float, target: float) -> bool:
    """True when the day's gain has reached ``target`` (a fraction).

    A ``target`` of 0 or less disables the lock and returns False.
    """
    if target is None or target <= 0:
        return False
    return daily_gain(daily_start, equity) >= target
