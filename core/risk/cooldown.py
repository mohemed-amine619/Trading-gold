"""
core/risk/cooldown.py — a pause on one direction after a losing trade.

Why: a stop-out in one direction usually means the market is not doing what the
signal assumed. A report showed 45 entries opened within 15 minutes of the
previous one in the same direction, and a 12-trade losing streak. The pause is
per direction only: a signal the other way is not delayed.

Time is counted in CLOSED BARS, supplied by the caller as an increasing counter,
so this class needs no clock and is trivial to test. ``bars = 0`` disables it.
"""

from __future__ import annotations

from typing import Dict


class DirectionCooldown:
    def __init__(self) -> None:
        self._until: Dict[int, int] = {}

    def record_loss(self, direction: int, bar: int, bars: int) -> None:
        """A losing trade in ``direction`` ended at bar number ``bar``: block that
        direction for the next ``bars`` bars (bars ``bar + 1 .. bar + bars``)."""
        if bars <= 0 or direction not in (1, -1):
            return
        self._until[direction] = max(self._until.get(direction, 0), bar + bars)

    def remaining(self, direction: int, bar: int) -> int:
        """Bars still blocked, counting ``bar`` itself (0 = free to trade)."""
        until = self._until.get(direction)
        return 0 if until is None else max(0, until - bar + 1)

    def blocked(self, direction: int, bar: int) -> bool:
        return self.remaining(direction, bar) > 0

    def clear(self) -> None:
        self._until.clear()
