"""
core/risk/daily_state.py — daily profit-target state machine (pure).

The bot is given a money target for the day (for example 70 USD). This module
decides what that target means for NEW risk and for the positions that are
already open, and nothing else. Everything it can ask for is on the safe side:
it never raises size to reach the target, never opens a trade, never widens a
stop. No martingale, no grid, no "chase the target".

    ACTIVE ──── net >= target ────▶ PROFIT_LOCK ── realized >= target ──┐
      │                                                                   ▼
      └─────────────────────── realized >= target ───────────▶ DAILY_TARGET_REACHED

Pure: no I/O, no clock reads, no broker import. The caller supplies the day,
the day's realised P&L and the floating P&L of the bot's open positions; the
module returns a ``DayDecision`` describing what the caller may do.

Semantics (each one is covered by a test)
-----------------------------------------
* ``target_usd`` of ``None`` or ``<= 0`` disables everything: the decision is
  always ACTIVE, ``allow_entries`` True, ``sweep`` None, whatever the inputs.
  A target that is NaN, infinite or not a number is a configuration bug and
  raises ``ValueError`` (it is never silently read as "disabled").
* A new ``day`` value resets the machine to ACTIVE before anything is
  evaluated. The day boundary is whatever the caller says (a ``date``, an ISO
  string, an int); this module invents no session hours and reads no clock. A
  ``datetime`` is truncated to its calendar date.
* ``realized`` or ``floating`` that is None, NaN, infinite or not a real number
  is UNKNOWN. It is never read as 0 and never as profit. An unknown input
  leaves the state unchanged, ``changed`` False, and FAILS CLOSED for that
  tick: ``allow_entries`` False, reason ``"daily pnl unknown"``, and no sweep
  (nothing is closed or tightened on a number that cannot be trusted; stops
  already moved at the broker stay where they are). The next tick with good
  data recovers. With the target disabled unknown inputs are irrelevant.
* ``net = realized + floating``. ACTIVE -> PROFIT_LOCK when ``net >= target``.
  ACTIVE or PROFIT_LOCK -> DAILY_TARGET_REACHED when ``realized >= target``
  (the target is banked; floating P&L no longer matters). States only move
  FORWARD within a day: they are latched and never go back, even when net later
  falls below the target. The day is done; the lock protects the day, it does
  not re-arm. A comparison tolerance of 1e-9 USD absorbs float noise at the
  boundary.
* ``allow_entries`` is True only in ACTIVE (and when disabled).
* ``sweep``: while PROFIT_LOCK the decision carries the configured sweep mode
  (``"ratchet"`` or ``"close_all"``) on EVERY known-data update. The caller's
  actions are idempotent (a ratchet only tightens; closing an already closed
  position is a no-op). In ACTIVE and DAILY_TARGET_REACHED it is None.
* ``lock_floor_usd`` is the profit the sweep must preserve from the open
  positions: ``max(0, target + lock_buffer_usd - realized)`` while PROFIT_LOCK,
  else 0. ``lock_buffer_usd`` is a cushion for spread, commission and slippage
  on top of the target.
* ``remaining = max(0, target - net)``; 0 when disabled; ``None`` when the
  target is enabled but ``net`` is unknown. It is ``None`` rather than NaN so a
  JSON status endpoint cannot be broken by it.
* ``changed`` is True only on the ``update()`` call in which the latched state
  differs from the state before the call (a day-rollover reset counts).
  Disabling or enabling the target through ``configure()`` is not a transition.
* ``configure()`` updates parameters live WITHOUT resetting the day's state.
  Raising the target does NOT un-latch a state already reached. Lowering it to
  ``<= 0`` disables the feature; the latch is frozen, not cleared, so
  re-enabling on the same day resumes it. Only ``reset()`` or a new day clears
  it. An invalid ``sweep_mode`` raises ``ValueError`` and changes nothing.
* ``snapshot()`` / ``restore()`` let a caller survive a restart. A snapshot only
  applies to the same day and never moves a state backwards within that day.

Sizing and sweep helpers (pure functions below the class)
----------------------------------------------------------
``ratchet_keep_fraction`` -> ``plan_sweep`` -> ``ratchet_stop`` turn a lock floor
into a stop that only ever tightens. ``target_size_factor`` scales risk DOWN by
confidence and by the profit left to the target and is never above 1.0.
``bot_floating_pnl`` sums the bot's own floating P&L (magic-number filtered).
"""

from __future__ import annotations

import math
import numbers
import threading
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = [
    "DayState",
    "DayDecision",
    "DailyTargetController",
    "SweepPlan",
    "SWEEP_RATCHET",
    "SWEEP_CLOSE_ALL",
    "SWEEP_MODES",
    "DEFAULT_MAX_KEEP",
    "finite_or_none",
    "day_key",
    "ratchet_keep_fraction",
    "plan_sweep",
    "ratchet_stop",
    "target_size_factor",
    "bot_floating_pnl",
]

SWEEP_RATCHET = "ratchet"
SWEEP_CLOSE_ALL = "close_all"
SWEEP_MODES = (SWEEP_RATCHET, SWEEP_CLOSE_ALL)

# A stop that keeps more than this share of the floating profit sits so close to
# the market that it is a market exit in all but name: the caller closes instead.
DEFAULT_MAX_KEEP = 0.85

# USD. Absorbs float noise when a sum lands a hair under the target.
_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Small pure helpers
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


def _coerce_target(value: Any) -> Optional[float]:
    """Target in USD, or ``None`` for "disabled". Garbage raises ValueError."""
    if value is None:
        return None
    number = finite_or_none(value)
    if number is None:
        raise ValueError(
            f"target_usd must be a finite number or None, got {value!r}"
        )
    return number if number > 0.0 else None


def _coerce_buffer(value: Any) -> float:
    if value is None:
        return 0.0
    number = finite_or_none(value)
    if number is None or number < 0.0:
        raise ValueError(
            f"lock_buffer_usd must be a finite number >= 0, got {value!r}"
        )
    return number


def _coerce_mode(value: Any) -> str:
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in SWEEP_MODES:
            return mode
    raise ValueError(f"sweep_mode must be one of {SWEEP_MODES}, got {value!r}")


class DayState(str, Enum):
    """Where the day stands against its profit target. Ordered, latched."""

    ACTIVE = "ACTIVE"
    PROFIT_LOCK = "PROFIT_LOCK"
    DAILY_TARGET_REACHED = "DAILY_TARGET_REACHED"


_RANK = {
    DayState.ACTIVE: 0,
    DayState.PROFIT_LOCK: 1,
    DayState.DAILY_TARGET_REACHED: 2,
}


@dataclass(frozen=True)
class DayDecision:
    """What the caller may do after one ``update()``.

    ``realized``, ``floating`` and ``net`` are ``None`` when unknown. ``sweep``
    is ``None``, ``"ratchet"`` or ``"close_all"``. ``changed`` is True only on
    the call in which the latched state changed.
    """

    state: DayState
    allow_entries: bool
    sweep: Optional[str]
    realized: Optional[float]
    floating: Optional[float]
    net: Optional[float]
    remaining: Optional[float]
    lock_floor_usd: float
    reason: str
    changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Plain, JSON-safe view (state as its string value)."""
        return {
            "state": self.state.value,
            "allow_entries": self.allow_entries,
            "sweep": self.sweep,
            "realized": self.realized,
            "floating": self.floating,
            "net": self.net,
            "remaining": self.remaining,
            "lock_floor_usd": self.lock_floor_usd,
            "reason": self.reason,
            "changed": self.changed,
        }


_UNSET: Any = object()


# ─────────────────────────────────────────────────────────────────────────────
# The state machine
# ─────────────────────────────────────────────────────────────────────────────

class DailyTargetController:
    """Latched ACTIVE -> PROFIT_LOCK -> DAILY_TARGET_REACHED machine.

    Thread-safe (one lock around every method): the trading loop calls
    ``update()`` while a dashboard thread may call ``configure()``.

    Args:
        target_usd: Profit target for the day in account currency. ``None`` or
            ``<= 0`` disables the feature.
        lock_buffer_usd: Extra profit, on top of the target, that the sweep
            preserves while PROFIT_LOCK (spread, commission, slippage cushion).
        sweep_mode: ``"ratchet"`` (tighten stops to keep the lock, close what
            cannot be protected) or ``"close_all"`` (bank everything).
    """

    def __init__(
        self,
        target_usd: Optional[float],
        lock_buffer_usd: Optional[float] = 0.0,
        sweep_mode: str = SWEEP_RATCHET,
    ) -> None:
        self._lock = threading.Lock()
        self._target: Optional[float] = _coerce_target(target_usd)
        self._buffer: float = _coerce_buffer(lock_buffer_usd)
        self._mode: str = _coerce_mode(sweep_mode)
        self._state: DayState = DayState.ACTIVE
        self._day_key: Optional[str] = None
        self._day_raw: Any = None

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def state(self) -> DayState:
        """The latched state (what ``snapshot()`` would persist)."""
        with self._lock:
            return self._state

    @property
    def day(self) -> Any:
        """The day value last seen, exactly as the caller passed it (or None)."""
        with self._lock:
            return self._day_raw

    @property
    def target_usd(self) -> Optional[float]:
        """The active target, or ``None`` when the feature is disabled."""
        with self._lock:
            return self._target

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._target is not None

    @property
    def lock_buffer_usd(self) -> float:
        with self._lock:
            return self._buffer

    @property
    def sweep_mode(self) -> str:
        with self._lock:
            return self._mode

    # ── configuration ───────────────────────────────────────────────────────

    def configure(
        self,
        target_usd: Any = _UNSET,
        lock_buffer_usd: Any = _UNSET,
        sweep_mode: Any = _UNSET,
    ) -> None:
        """Change parameters live. Omitted arguments keep their value.

        Never resets the day's state. Raising the target does not un-latch a
        state already reached; lowering it to ``<= 0`` (or ``None``) disables
        the feature while keeping the latch frozen. All arguments are validated
        before any is applied, so a bad ``sweep_mode`` leaves everything as it
        was. Raises ``ValueError`` on an invalid value.
        """
        new_target = _coerce_target(target_usd) if target_usd is not _UNSET else _UNSET
        new_buffer = _coerce_buffer(lock_buffer_usd) if lock_buffer_usd is not _UNSET else _UNSET
        new_mode = _coerce_mode(sweep_mode) if sweep_mode is not _UNSET else _UNSET
        with self._lock:
            if new_target is not _UNSET:
                self._target = new_target
            if new_buffer is not _UNSET:
                self._buffer = new_buffer
            if new_mode is not _UNSET:
                self._mode = new_mode

    # ── day control ─────────────────────────────────────────────────────────

    def reset(self, day: Any = None) -> None:
        """Back to ACTIVE, optionally pinning the current day.

        The explicit manual override (the equivalent of the dashboard's "reset
        halt"). Without it only a new day clears a latched state.
        """
        with self._lock:
            self._state = DayState.ACTIVE
            self._day_key = day_key(day)
            self._day_raw = day if self._day_key is not None else None

    def snapshot(self) -> Dict[str, Any]:
        """JSON-safe ``{"day": <key or None>, "state": <state value>}``.

        Rows returned by ``daily_pnl.latest_day_event`` carry the same two keys
        and are valid input for ``restore``.
        """
        with self._lock:
            return {"day": self._day_key, "state": self._state.value}

    def restore(self, snapshot: Any, day: Any) -> bool:
        """Re-apply a persisted state. Returns True when it was applied.

        Applies only when the snapshot is for the same ``day`` as the argument,
        and never moves a state backwards within that day (a stale snapshot
        cannot un-latch a lock that is already in force). Anything malformed is
        ignored and returns False. This method does not raise.
        """
        try:
            key = day_key(day)
            if key is None or not isinstance(snapshot, Mapping):
                return False
            if day_key(snapshot.get("day")) != key:
                return False
            raw_state = snapshot.get("state")
            state = (raw_state if isinstance(raw_state, DayState)
                     else DayState(str(raw_state).strip().upper()))
        except (ValueError, TypeError):
            return False
        with self._lock:
            if self._day_key == key and _RANK[state] < _RANK[self._state]:
                return False
            self._day_key = key
            self._day_raw = day
            self._state = state
        return True

    # ── the one operation that matters ──────────────────────────────────────

    def update(self, day: Any, realized: Any, floating: Any) -> DayDecision:
        """Evaluate one tick. Never raises; see the module docstring."""
        with self._lock:
            before = self._state
            key = day_key(day)
            rolled = False
            if key is not None and key != self._day_key:
                rolled = self._state is not DayState.ACTIVE
                self._state = DayState.ACTIVE
                self._day_key = key
                self._day_raw = day

            r = finite_or_none(realized)
            f = finite_or_none(floating)
            net = (r + f) if (r is not None and f is not None) else None
            target = self._target
            prefix = f"new day {key}: reset to ACTIVE; " if rolled else ""

            if target is None:
                return self._decision(
                    before, DayState.ACTIVE, True, None, r, f, net, 0.0, 0.0,
                    prefix + "daily target disabled",
                )

            if key is None:
                return self._decision(
                    before, self._state, False, None, r, f, net,
                    None if net is None else max(0.0, target - net), 0.0,
                    "daily day unknown: cannot attribute pnl to a day",
                )

            if r is None or f is None or net is None:
                held = "" if self._state is DayState.ACTIVE else f" ({self._state.value} held)"
                return self._decision(
                    before, self._state, False, None, r, f, None, None, 0.0,
                    prefix + "daily pnl unknown" + held,
                )

            # Known data: advance the latch (forward only).
            start = self._state
            if r + _EPS >= target:
                candidate = DayState.DAILY_TARGET_REACHED
            elif net + _EPS >= target:
                candidate = DayState.PROFIT_LOCK
            else:
                candidate = DayState.ACTIVE
            if _RANK[candidate] > _RANK[self._state]:
                self._state = candidate

            state = self._state
            remaining = max(0.0, target - net)
            advanced = state is not start
            if state is DayState.ACTIVE:
                reason = (f"below target: net {net:+.2f} of {target:.2f} "
                          f"({remaining:.2f} to go)")
                return self._decision(before, state, True, None, r, f, net,
                                      remaining, 0.0, prefix + reason)
            if state is DayState.PROFIT_LOCK:
                floor = max(0.0, target + self._buffer - r)
                if advanced:
                    head = f"profit lock engaged: net {net:+.2f} >= target {target:.2f}"
                else:
                    head = f"profit lock held (latched): net {net:+.2f} vs target {target:.2f}"
                reason = (f"{head}; entries blocked, sweep={self._mode}, "
                          f"preserve {floor:.2f}")
                return self._decision(before, state, False, self._mode, r, f, net,
                                      remaining, floor, prefix + reason)
            if advanced:
                head = f"daily target reached: realized {r:+.2f} >= target {target:.2f}"
            else:
                head = f"daily target held (latched): realized {r:+.2f} vs target {target:.2f}"
            reason = f"{head}; entries blocked for the day"
            return self._decision(before, state, False, None, r, f, net,
                                  remaining, 0.0, prefix + reason)

    def _decision(
        self,
        before: DayState,
        state: DayState,
        allow: bool,
        sweep: Optional[str],
        realized: Optional[float],
        floating: Optional[float],
        net: Optional[float],
        remaining: Optional[float],
        floor: float,
        reason: str,
    ) -> DayDecision:
        return DayDecision(
            state=state,
            allow_entries=allow,
            sweep=sweep,
            realized=realized,
            floating=floating,
            net=net,
            remaining=remaining,
            lock_floor_usd=floor,
            reason=reason,
            changed=self._state is not before,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sweep planning: turn a lock floor into stops that only ever tighten
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SweepPlan:
    """How to protect the lock. ``keep_fraction`` is None for ``close_all``."""

    action: str
    keep_fraction: Optional[float]
    reason: str


def ratchet_keep_fraction(
    lock_floor_usd: Any,
    floating_usd: Any,
    max_keep: Any = DEFAULT_MAX_KEEP,
) -> Optional[float]:
    """Share of the CURRENT floating profit a trailing stop must keep.

    ``realized + kept >= target + buffer`` holds when the stop keeps
    ``lock_floor_usd / floating_usd`` of the floating profit. ``None`` means "a
    stop cannot do that": the needed share exceeds ``max_keep`` (a stop that
    tight is a market exit anyway), the floating P&L is not positive, or any
    input is unknown. The caller must then close instead. A floor ``<= 0``
    needs nothing preserved and returns 0.0.
    """
    floor = finite_or_none(lock_floor_usd)
    floating = finite_or_none(floating_usd)
    cap = finite_or_none(max_keep)
    if floor is None or floating is None or cap is None:
        return None
    if floor <= 0.0:
        return 0.0
    if floating <= 0.0:
        return None
    needed = floor / floating
    if not math.isfinite(needed) or needed > min(cap, 1.0):
        return None
    return needed


def plan_sweep(
    mode: Any,
    lock_floor_usd: Any,
    position_floats: Optional[Iterable[Any]],
    max_keep: Any = DEFAULT_MAX_KEEP,
) -> SweepPlan:
    """Choose between tightening stops and banking everything.

    ``position_floats`` are the floating P&L of ALL the bot's positions
    (account wide, not one symbol). ``"ratchet"`` is chosen only when there is at
    least one position, every float is finite and strictly positive (a loser or
    a break-even position cannot be protected by a stop, so everything is
    banked instead) and ``ratchet_keep_fraction`` of their sum exists.
    Otherwise the plan is ``close_all`` with the reason. Never raises: an
    unrecognised mode also banks everything, the most protective action.
    """
    name = mode.strip().lower() if isinstance(mode, str) else None
    if name == SWEEP_CLOSE_ALL:
        return SweepPlan(SWEEP_CLOSE_ALL, None, "sweep mode is close_all")
    if name != SWEEP_RATCHET:
        return SweepPlan(SWEEP_CLOSE_ALL, None,
                         f"unrecognised sweep mode {mode!r}: banking everything")

    try:
        values = [] if position_floats is None else [finite_or_none(v) for v in position_floats]
    except (TypeError, ValueError):
        return SweepPlan(SWEEP_CLOSE_ALL, None, "position pnl unreadable: banking everything")
    if not values:
        return SweepPlan(SWEEP_CLOSE_ALL, None, "no open bot positions to protect")
    if any(v is None for v in values):
        return SweepPlan(SWEEP_CLOSE_ALL, None, "a position pnl is unknown: banking everything")
    if any(v <= 0.0 for v in values):
        return SweepPlan(SWEEP_CLOSE_ALL, None,
                         "a position is not in profit, a stop cannot protect it: banking everything")

    total = math.fsum(values)
    keep = ratchet_keep_fraction(lock_floor_usd, total, max_keep)
    if keep is None:
        return SweepPlan(
            SWEEP_CLOSE_ALL, None,
            f"lock floor {finite_or_none(lock_floor_usd)} cannot be kept by a stop on "
            f"{total:.2f} floating: banking everything",
        )
    return SweepPlan(SWEEP_RATCHET, keep, f"ratchet: keep {keep:.1%} of {total:.2f} floating")


def ratchet_stop(
    direction: Any,
    entry: Any,
    price: Any,
    current_sl: Any,
    keep_fraction: Any,
    min_gap: Any = 0.0,
) -> Optional[float]:
    """A tighter stop that keeps ``keep_fraction`` of the open profit, or None.

    The candidate stop is ``entry + keep_fraction * (price - entry)`` (the same
    expression serves a long and a short), then pulled at least ``min_gap``
    BEHIND ``price`` (``min_gap`` is the broker's minimum stop distance in price
    units, supplied by the caller; ``price`` is the side of the book that
    triggers the stop: bid for a long, ask for a short). It is returned ONLY if
    it strictly tightens ``current_sl`` (``0`` or ``None`` means no stop yet).

    This is the safety-critical rule, in the spirit of ``core.position.trailing``:
    for NO input does the function return a stop looser than ``current_sl``.
    It returns None, never a guess, when anything is unknown (NaN, inf, None,
    wrong type), when ``keep_fraction`` is outside ``[0, 1]``, when ``min_gap`` is
    negative, or when the position is not in profit (a stop cannot protect a
    loser; the caller closes instead). The result is not snapped to the broker's
    price grid, and a ``min_gap`` clamp can make it tighter than the share asked
    for; a caller that must guarantee the floor compares and closes instead.
    """
    side = finite_or_none(direction)
    if side not in (1.0, -1.0):
        return None
    entry_px = finite_or_none(entry)
    last_px = finite_or_none(price)
    keep = finite_or_none(keep_fraction)
    gap = finite_or_none(min_gap)
    if entry_px is None or last_px is None or keep is None or gap is None:
        return None
    if not 0.0 <= keep <= 1.0 or gap < 0.0:
        return None

    stop: Optional[float] = None
    if current_sl is not None:
        stop = finite_or_none(current_sl)
        if stop is None:
            return None  # an unreadable current stop: cannot prove a tightening
        if stop == 0.0:
            stop = None  # MT5 convention: 0 means "no stop"

    if not (last_px - entry_px) * side > 0.0:
        return None  # not in profit

    candidate = entry_px + keep * (last_px - entry_px)
    if side > 0:
        new = min(candidate, last_px - gap)
        if not new < last_px:
            return None
        if stop is not None and not new > stop:
            return None
    else:
        new = max(candidate, last_px + gap)
        if not new > last_px:
            return None
        if stop is not None and not new < stop:
            return None
    return new if math.isfinite(new) else None


def target_size_factor(
    confidence: Any,
    min_confidence: Any,
    remaining_usd: Any,
    risk_budget_usd: Any,
    rr: Any,
    floor: Any = 0.25,
    conf_floor: Any = 0.5,
) -> float:
    """Risk multiplier in ``[floor, 1.0]``: scale DOWN, never up.

    ``confidence_factor`` rises from ``conf_floor`` at ``min_confidence`` to 1.0
    at full confidence. ``remaining_factor`` is ``remaining_usd / (risk_budget_usd
    * rr)`` clipped to ``[floor, 1]``: when little of the target is left a full
    size trade would risk more than its win is worth, so size shrinks. The
    result is their product clipped to ``[floor, 1.0]``.

    * Unknown confidence (None/NaN) counts as AT the threshold (``conf_floor``):
      uncertainty never upsizes.
    * Unknown or non-positive ``remaining_usd``, ``risk_budget_usd`` or ``rr``
      (target disabled, or inputs missing) give ``remaining_factor = 1.0``:
      no reduction, and never an increase.
    * Being behind the target, or losing money today, makes ``remaining_usd``
      large and so cannot raise the factor above 1.0. This is the opposite of
      martingale.
    * Monotone: higher confidence never lowers the factor; over valid positive
      inputs, more remaining profit never lowers it.
    """
    low = _unit(floor, 0.25)
    conf_low = _unit(conf_floor, 0.5)

    conf = finite_or_none(confidence)
    threshold = finite_or_none(min_confidence)
    if conf is None or threshold is None or (1.0 - threshold) <= 0.0:
        confidence_factor = conf_low
    else:
        x = (conf - threshold) / (1.0 - threshold)
        if x != x:  # NaN from an overflowing garbage input
            x = 0.0
        x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
        confidence_factor = conf_low + (1.0 - conf_low) * x

    remaining = finite_or_none(remaining_usd)
    budget = finite_or_none(risk_budget_usd)
    ratio = finite_or_none(rr)
    if (remaining is not None and budget is not None and ratio is not None
            and remaining > 0.0 and budget > 0.0 and ratio > 0.0):
        denominator = budget * ratio
        # A denominator that underflows to 0 means a win is worth nothing per
        # unit of remaining profit: the whole remainder is "enough", no reduction.
        remaining_factor = remaining / denominator if denominator > 0.0 else 1.0
        if remaining_factor != remaining_factor:
            remaining_factor = 1.0
        remaining_factor = low if remaining_factor < low else 1.0 if remaining_factor > 1.0 else remaining_factor
    else:
        remaining_factor = 1.0

    result = confidence_factor * remaining_factor
    if result != result:
        return low
    return low if result < low else 1.0 if result > 1.0 else result


def _unit(value: Any, default: float) -> float:
    """``value`` clamped to ``[0, 1]``; ``default`` when it is not a number."""
    number = finite_or_none(value)
    if number is None:
        return default
    return 0.0 if number < 0.0 else 1.0 if number > 1.0 else number


def bot_floating_pnl(positions: Optional[Iterable[Any]], magic: Any) -> Optional[float]:
    """Floating P&L of the bot's own positions, or ``None`` when unknown.

    Sums ``profit + swap`` (plus ``commission`` when the attribute exists) over
    the objects whose ``.magic`` equals ``magic``; other magic numbers (manual
    trades) are ignored, and an object with no ``.magic`` counts as magic 0, as
    ``run.py`` does. A missing or ``None`` ``swap``/``commission`` counts as 0,
    but a matching position whose ``profit`` is missing or not finite (or whose
    ``swap``/``commission`` is present but not finite) makes the result
    ``None``: the caller can refuse to act on a number it cannot trust. No
    matching position is a genuine 0.0; ``positions`` itself being ``None`` (a
    failed broker call) or ``magic`` being unusable is unknown, not zero.
    """
    wanted = finite_or_none(magic)
    if positions is None or wanted is None:
        return None
    try:
        items = list(positions)
    except TypeError:
        return None
    terms = []
    for pos in items:
        have = finite_or_none(getattr(pos, "magic", 0))
        if have is None or have != wanted:
            continue
        profit = finite_or_none(getattr(pos, "profit", None))
        if profit is None:
            return None
        terms.append(profit)
        for extra in ("swap", "commission"):
            raw = getattr(pos, extra, None)
            if raw is None:
                continue
            value = finite_or_none(raw)
            if value is None:
                return None
            terms.append(value)
    try:
        total = math.fsum(terms)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None
