"""
core/risk/streak.py — losing-streak size reduction (anti-martingale).

After a run of consecutive losing trades the next entries are sized down.
Size is never increased after a loss, and never raised above the base size:
the multiplier is always in ``(0, 1]``. That makes this an anti-martingale
brake, which AGENTS.md permits, and not a recovery ramp, which it forbids.

Pure function, no state. The caller supplies the recent closed-trade P&L,
oldest first, and gets back a multiplier to apply to the base position size.
"""

from __future__ import annotations

from typing import Sequence

# A P&L inside this band counts as breakeven: it neither extends a losing
# streak nor ends one with a win.
_EPS = 1e-9


def streak_multiplier(pnls: Sequence[float], losses_to_halve: int = 2,
                      factor: float = 0.5) -> float:
    """Return the size multiplier implied by the most recent trades.

    Counts consecutive losses from the newest trade backwards. A breakeven or
    winning trade stops the count. When the count reaches ``losses_to_halve``
    the multiplier is ``factor``; otherwise it is 1.0.
    """
    if losses_to_halve < 1:
        raise ValueError("losses_to_halve must be at least 1")
    if not (0.0 < factor <= 1.0):
        raise ValueError("factor must be in (0, 1]")

    losses = 0
    for pnl in reversed(pnls):
        if pnl >= -_EPS:
            break
        losses += 1
    return factor if losses >= losses_to_halve else 1.0
