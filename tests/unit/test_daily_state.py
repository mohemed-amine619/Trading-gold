"""The daily profit-target state machine: latch, lock actions, reset, no chasing."""

from core.risk.daily_state import (
    DailyTargetTracker, DayState, LockAction, LOCK_CLOSE, LOCK_RATCHET,
)

D1, D2 = "2026-11-04", "2026-11-05"


def test_disabled_target_is_always_active():
    t = DailyTargetTracker(0.0)
    d = t.update(D1, 500.0, 500.0, 2)
    assert d.state is DayState.ACTIVE and d.entries_allowed and d.action is LockAction.NONE


def test_below_target_stays_active_and_entries_allowed():
    t = DailyTargetTracker(70.0, 5.0)
    d = t.update(D1, 40.0, 20.0, 1)
    assert d.state is DayState.ACTIVE and d.entries_allowed and d.net == 60.0


def test_net_includes_floating_and_triggers_lock_close():
    t = DailyTargetTracker(70.0, 5.0, LOCK_CLOSE)
    d = t.update(D1, 50.0, 20.0, 2)
    assert d.state is DayState.LOCKING
    assert d.action is LockAction.CLOSE_ALL
    assert not d.entries_allowed and d.transitioned


def test_flat_after_lock_becomes_reached_and_latches_even_below_target():
    t = DailyTargetTracker(70.0, 5.0)
    t.update(D1, 50.0, 20.0, 2)
    # Closing slippage left the realized net a hair under the target.
    d = t.update(D1, 69.5, 0.0, 0)
    assert d.state is DayState.DAILY_TARGET_REACHED
    assert not d.entries_allowed and d.transitioned
    # Still latched on later ticks, even if equity drifts.
    d = t.update(D1, 60.0, 0.0, 0)
    assert d.state is DayState.DAILY_TARGET_REACHED and not d.transitioned


def test_transition_is_reported_once():
    t = DailyTargetTracker(70.0)
    assert t.update(D1, 80.0, 0.0, 1).transitioned
    assert not t.update(D1, 80.0, 0.0, 1).transitioned


def test_ratchet_mode_tightens_then_closes_when_buffer_given_back():
    t = DailyTargetTracker(70.0, 5.0, LOCK_RATCHET)
    d = t.update(D1, 60.0, 12.0, 1)                 # net 72
    assert d.state is DayState.LOCKING and d.action is LockAction.RATCHET
    d = t.update(D1, 60.0, 6.0, 1)                  # net 66, floor 65 -> still ratchet
    assert d.action is LockAction.RATCHET
    d = t.update(D1, 60.0, 3.0, 1)                  # net 63 < floor 65
    assert d.action is LockAction.CLOSE_ALL


def test_new_day_resets_the_latch():
    t = DailyTargetTracker(70.0)
    t.update(D1, 80.0, 0.0, 0)
    assert t.state is DayState.DAILY_TARGET_REACHED
    d = t.update(D2, 0.0, 0.0, 0)
    assert d.state is DayState.ACTIVE and d.entries_allowed


def test_restore_relatches_after_restart():
    t = DailyTargetTracker(70.0)
    t.restore(D1, "DAILY_TARGET_REACHED")
    d = t.update(D1, 0.0, 0.0, 0)
    assert d.state is DayState.DAILY_TARGET_REACHED and not d.entries_allowed


def test_restore_ignores_unknown_state():
    t = DailyTargetTracker(70.0)
    t.restore(D1, "garbage")
    assert t.state is DayState.ACTIVE


def test_buffer_is_clamped_to_the_target():
    t = DailyTargetTracker(70.0, 500.0)
    assert t.buffer == 70.0
    assert t.update(D1, 0.0, 0.0, 0).floor == 0.0


def test_a_losing_day_never_changes_the_machine_or_asks_for_more_risk():
    """No input of the machine can raise size: a deep loss simply stays ACTIVE."""
    t = DailyTargetTracker(70.0, 5.0)
    d = t.update(D1, -200.0, -10.0, 1)
    assert d.state is DayState.ACTIVE and d.action is LockAction.NONE
    assert not hasattr(d, "size_multiplier")
