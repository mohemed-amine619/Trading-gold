"""
core/risk/daily_state.py — the daily profit-target state machine.

    ACTIVE ──(net >= target)──> LOCKING ──(flat)──> DAILY_TARGET_REACHED
       ^                                                    │
       └────────────── new trading day (reset) ─────────────┘

``net`` is the day's realized P&L plus the floating P&L of the bot's own
positions, in account currency. Reaching the target ends the day's risk-taking;
it never creates risk. In particular this module has NO input that raises size
or trade count because the target is still far away: chasing a daily number by
sizing up after losses is martingale by another name, and that is banned.

While LOCKING the caller must bank the gain, and may only do so in ways that
cannot widen a stop:

* ``CLOSE_ALL``  close the bot's positions now (the default lock action), or
* ``RATCHET``    tighten stops only (the caller owns the tightening, which can
                 never loosen a stop) and let the trade run, falling back to
                 ``CLOSE_ALL`` if the net gives back more than ``buffer`` of
                 the target. The floor is ``target - buffer``.

Once triggered the day is latched: it cannot go back to ACTIVE that day even if
closing slippage leaves the net a few cents under the target.

A target of 0 or less disables the machine (always ACTIVE). Pure and
deterministic apart from the tracker's own latch; no broker or database import.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Hashable, Optional


class DayState(str, Enum):
    ACTIVE = "ACTIVE"
    LOCKING = "LOCKING"
    DAILY_TARGET_REACHED = "DAILY_TARGET_REACHED"


class LockAction(str, Enum):
    NONE = "NONE"
    CLOSE_ALL = "CLOSE_ALL"
    RATCHET = "RATCHET"


LOCK_CLOSE = "close"
LOCK_RATCHET = "ratchet"


@dataclass(frozen=True)
class DayDecision:
    state: DayState
    action: LockAction
    net: float                    # realized + floating, bot positions only
    target: float
    floor: float                  # net below this while ratcheting forces CLOSE_ALL
    entries_allowed: bool
    transitioned: bool            # True only on the update that changed the state
    reason: str


class DailyTargetTracker:
    """Latched day state. Feed it every tick; it resets on a new ``day_key``."""

    def __init__(self, target_usd: float = 0.0, buffer_usd: float = 0.0,
                 lock_action: str = LOCK_CLOSE):
        self.configure(target_usd, buffer_usd, lock_action)
        self._day: Optional[Hashable] = None
        self._state = DayState.ACTIVE

    def configure(self, target_usd: float, buffer_usd: float,
                  lock_action: str = LOCK_CLOSE) -> None:
        self.target = max(0.0, float(target_usd or 0.0))
        # The buffer can never push the floor below zero or above the target.
        self.buffer = min(max(0.0, float(buffer_usd or 0.0)), self.target)
        self.lock_action = LOCK_RATCHET if lock_action == LOCK_RATCHET else LOCK_CLOSE

    @property
    def state(self) -> DayState:
        return self._state

    def restore(self, day_key: Hashable, state: str) -> None:
        """Re-latch after a restart so a reboot cannot reopen a finished day."""
        self._day = day_key
        try:
            self._state = DayState(state)
        except ValueError:
            self._state = DayState.ACTIVE

    def update(self, day_key: Hashable, realized: float, floating: float,
               n_open: int) -> DayDecision:
        if day_key != self._day:                       # new trading day
            self._day = day_key
            self._state = DayState.ACTIVE
        net = float(realized) + float(floating)
        floor = self.target - self.buffer
        prev = self._state

        if self.target <= 0:
            self._state = DayState.ACTIVE
            return DayDecision(DayState.ACTIVE, LockAction.NONE, net, 0.0, 0.0, True,
                               prev is not DayState.ACTIVE, "daily target disabled")

        if prev is DayState.ACTIVE and net >= self.target:
            self._state = DayState.LOCKING
        if self._state is DayState.LOCKING and n_open <= 0:
            self._state = DayState.DAILY_TARGET_REACHED
        state = self._state
        changed = state is not prev

        if state is DayState.ACTIVE:
            return DayDecision(state, LockAction.NONE, net, self.target, floor, True, changed,
                               f"net {net:+.2f} of {self.target:.2f}")
        if state is DayState.DAILY_TARGET_REACHED:
            return DayDecision(state, LockAction.NONE, net, self.target, floor, False, changed,
                               f"daily target {self.target:.2f} reached (net {net:+.2f}): "
                               "no new entries today")
        # LOCKING: entries are vetoed and the gain must be banked.
        if self.lock_action == LOCK_CLOSE:
            action = LockAction.CLOSE_ALL
        elif net < floor:
            action = LockAction.CLOSE_ALL              # gave back more than the buffer
        else:
            action = LockAction.RATCHET
        return DayDecision(state, action, net, self.target, floor, False, changed,
                           f"net {net:+.2f} reached target {self.target:.2f}: "
                           f"{action.value.lower().replace('_', ' ')}")


# ─────────────────────────────────────────────────────────────────────────────
# Small pure helpers (used by core/risk/daily_pnl.py)
# ─────────────────────────────────────────────────────────────────────────────

def finite_or_none(value: Any) -> Optional[float]:
    """``value`` as a finite ``float``, or ``None`` when it is unknown.

    None, NaN, +/-inf, booleans, strings and anything that is not a real number
    are UNKNOWN. They are never converted to 0.0: a missing P&L is not a flat
    day, and a missing price is not a price.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, numbers.Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def day_key(day: Any) -> Optional[str]:
    """Canonical text key for a "day" value, or ``None`` when it is unusable.

    ``date`` -> ISO date; ``datetime`` -> the ISO date it falls on (passing
    ``datetime.now()`` by mistake must not reset the machine on every tick);
    ``str`` -> itself, stripped; integers -> their decimal text. Booleans,
    floats and every other type are rejected: a float such as ``time.time()``
    would change on each call and silently disarm the latch.
    """
    if day is None or isinstance(day, bool):
        return None
    try:
        if isinstance(day, datetime):
            text = day.date().isoformat()
            return None if text == "NaT" else text  # pandas NaT is a datetime
        if isinstance(day, date):
            return day.isoformat()
    except (ValueError, AttributeError):  # pragma: no cover - defensive
        return None
    if isinstance(day, numbers.Integral):
        return str(int(day))
    if isinstance(day, str):
        text = day.strip()
        return text or None
    return None
