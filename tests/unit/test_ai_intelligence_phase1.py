"""
tests/unit/test_ai_intelligence_phase1.py — pure logic behind the phase-1 AI
and risk changes: purged validation split, first-barrier labels and losing-
streak sizing.

These are pure functions in core/, so they run on Linux with no MT5. The
run.py wiring is covered in test_phase1_wiring.py and test_ai_model_rigor.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.ai.labels import first_barrier_labels
from core.ai.validation import purged_split
from core.risk.streak import streak_multiplier


# ─────────────────────────────────────────────────────────────────────────────
# purged_split
# ─────────────────────────────────────────────────────────────────────────────

def test_purged_split_leaves_a_gap_equal_to_the_horizon():
    train, val = purged_split(100, n_val=20, gap=5)
    assert train == slice(0, 75)
    assert val == slice(80, 100)
    # Rows 75..79 are dropped: the last training label must not reach val.
    assert (val.start - train.stop) == 5


def test_purged_split_with_zero_gap_matches_the_old_split():
    train, val = purged_split(100, n_val=20, gap=0)
    assert train == slice(0, 80)
    assert val == slice(80, 100)


def test_purged_split_never_overlaps():
    train, val = purged_split(500, n_val=100, gap=12)
    assert train.stop <= val.start


@pytest.mark.parametrize("n_val, gap", [(0, 5), (-3, 5), (10, -1)])
def test_purged_split_rejects_nonsense_arguments(n_val, gap):
    with pytest.raises(ValueError):
        purged_split(100, n_val=n_val, gap=gap)


def test_purged_split_rejects_a_window_with_no_training_rows():
    with pytest.raises(ValueError):
        purged_split(30, n_val=25, gap=5)


# ─────────────────────────────────────────────────────────────────────────────
# first_barrier_labels
# ─────────────────────────────────────────────────────────────────────────────

def _flat(n, price=100.0):
    close = np.full(n, price)
    return close.copy(), close.copy(), close.copy(), np.ones(n)


def test_up_barrier_touched_first_labels_up():
    high, low, close, atr = _flat(4)
    high[1] = 101.5            # barrier = 1.3 ATR -> up at 101.3
    labels = first_barrier_labels(high, low, close, atr, 1.3, horizon=2)
    assert labels[0] == 1.0


def test_stop_side_touched_first_labels_down_even_if_price_later_rallies():
    """The case the old best-close label got wrong: stopped out, then rallied."""
    high, low, close, atr = _flat(5)
    low[1] = 98.0              # down barrier at 98.7 hit first
    high[2] = 102.0            # would have hit up barrier later
    labels = first_barrier_labels(high, low, close, atr, 1.3, horizon=3)
    assert labels[0] == 0.0


def test_timeout_inside_horizon_is_dropped():
    high, low, close, atr = _flat(10)
    labels = first_barrier_labels(high, low, close, atr, 1.3, horizon=3)
    assert np.isnan(labels[0])


def test_both_barriers_in_one_candle_is_ambiguous_and_dropped():
    high, low, close, atr = _flat(4)
    high[1] = 102.0
    low[1] = 98.0
    labels = first_barrier_labels(high, low, close, atr, 1.3, horizon=2)
    assert np.isnan(labels[0])


def test_last_horizon_bars_have_no_forward_data():
    high, low, close, atr = _flat(6)
    high[5] = 110.0
    labels = first_barrier_labels(high, low, close, atr, 1.3, horizon=2)
    assert all(np.isnan(labels[-2:]))


def test_bad_atr_produces_no_label():
    high, low, close, atr = _flat(4)
    high[1] = 110.0
    atr[0] = np.nan
    labels = first_barrier_labels(high, low, close, atr, 1.3, horizon=2)
    assert np.isnan(labels[0])


def test_barrier_scales_with_atr_not_a_fixed_point_count():
    """Multi-asset rule: the same move is a different label on a wider ATR."""
    high, low, close, atr = _flat(4)
    high[1] = 101.5
    small = first_barrier_labels(high, low, close, atr, 1.3, horizon=2)
    wide_atr = atr * 3.0       # barrier now 3.9 away, 101.5 no longer reaches it
    wide = first_barrier_labels(high, low, close, wide_atr, 1.3, horizon=2)
    assert small[0] == 1.0
    assert np.isnan(wide[0])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        first_barrier_labels([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 1.0], 1.3, 1)


@pytest.mark.parametrize("mult, horizon", [(0.0, 3), (-1.0, 3), (1.3, 0)])
def test_bad_parameters_are_refused(mult, horizon):
    high, low, close, atr = _flat(5)
    with pytest.raises(ValueError):
        first_barrier_labels(high, low, close, atr, mult, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# streak_multiplier
# ─────────────────────────────────────────────────────────────────────────────

def test_no_history_means_full_size():
    assert streak_multiplier([]) == 1.0


def test_one_loss_is_not_yet_a_streak():
    assert streak_multiplier([-1.0]) == 1.0


def test_two_consecutive_losses_halve_size():
    assert streak_multiplier([-1.0, -1.0]) == 0.5


def test_a_win_ends_the_streak():
    assert streak_multiplier([-1.0, -1.0, 5.0]) == 1.0


def test_only_the_most_recent_run_counts():
    assert streak_multiplier([5.0, -1.0, -1.0, -1.0]) == 0.5


def test_breakeven_ends_the_streak():
    assert streak_multiplier([-1.0, -1.0, 0.0]) == 1.0


def test_threshold_is_configurable():
    assert streak_multiplier([-1.0, -1.0], losses_to_halve=3) == 1.0
    assert streak_multiplier([-1.0, -1.0, -1.0], losses_to_halve=3) == 0.5


def test_multiplier_never_exceeds_base_size():
    """Anti-martingale: wins never upsize, losses never upsize."""
    assert streak_multiplier([5.0, 5.0, 5.0]) <= 1.0


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.5])
def test_factor_must_be_a_reduction(factor):
    with pytest.raises(ValueError):
        streak_multiplier([-1.0, -1.0], factor=factor)
