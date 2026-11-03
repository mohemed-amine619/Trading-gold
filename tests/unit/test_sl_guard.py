"""
tests/unit/test_sl_guard.py — the two rules every stop change must pass.

The loosening cases are the exact numbers from the live log:
  SELL 2164109663: stop 3905.048 then "Modified SL -> 3905.221"  (looser)
  SELL 2160013129: stop 3896.198 then "Modified SL -> 3896.433"  (looser)
"""

from __future__ import annotations

import pytest

from core.position.sl_guard import is_tighter, step_ok


class TestIsTighter:
    def test_sell_a_lower_stop_is_tighter(self):
        assert is_tighter(False, 3905.288, 3905.048) is True

    def test_sell_a_higher_stop_is_looser_the_live_log_bug(self):
        assert is_tighter(False, 3905.048, 3905.221) is False
        assert is_tighter(False, 3896.198, 3896.433) is False

    def test_buy_a_higher_stop_is_tighter(self):
        assert is_tighter(True, 3950.0, 3951.2) is True

    def test_buy_a_lower_stop_is_looser(self):
        assert is_tighter(True, 3951.2, 3950.0) is False

    @pytest.mark.parametrize("is_buy", [True, False])
    def test_an_identical_stop_is_not_tighter(self, is_buy):
        assert is_tighter(is_buy, 3900.0, 3900.0) is False

    @pytest.mark.parametrize("is_buy", [True, False])
    @pytest.mark.parametrize("current", [0, 0.0, None])
    def test_no_stop_set_means_any_stop_is_tighter(self, is_buy, current):
        assert is_tighter(is_buy, current, 3900.0) is True

    @pytest.mark.parametrize("is_buy", [True, False])
    @pytest.mark.parametrize("new", [None, 0, 0.0, -1.0])
    def test_a_missing_or_nonsense_new_stop_is_refused(self, is_buy, new):
        assert is_tighter(is_buy, 3900.0, new) is False
        assert is_tighter(is_buy, 0, new) is False

    def test_a_sub_epsilon_move_is_not_tighter(self):
        assert is_tighter(False, 3900.0, 3900.0 - 1e-12) is False


class TestStepOk:
    def test_a_large_enough_improvement_passes(self):
        assert step_ok(False, 3905.0, 3904.5, min_step=0.4) is True        # 0.5 >= 0.4

    def test_a_tiny_improvement_is_refused_by_the_step_floor(self):
        assert step_ok(False, 3905.0, 3904.99, min_step=0.4) is False      # the log's 0.01 drift
        assert step_ok(True, 3900.0, 3900.05, min_step=0.4) is False

    def test_exactly_the_step_passes(self):
        assert step_ok(False, 3905.0, 3904.6, min_step=0.4) is True

    def test_a_loosening_never_passes_even_with_a_zero_step(self):
        assert step_ok(False, 3905.048, 3905.221, min_step=0.0) is False
        assert step_ok(True, 3951.0, 3950.0, min_step=0.0) is False

    def test_no_stop_yet_passes_regardless_of_the_step(self):
        assert step_ok(False, 0, 3910.0, min_step=5.0) is True

    def test_the_first_move_to_breakeven_bypasses_the_step_floor(self):
        # SELL entered at 3900, break-even stop 3899.8. Stop is 3908 (initial).
        # Moving to 3899.8 is a state change even though 3908 -> 3899.8 is big;
        # the interesting case is a SMALL final move that reaches break-even:
        assert step_ok(False, 3899.95, 3899.8, min_step=0.4, breakeven_sl=3899.8) is True
        assert step_ok(True, 3900.05, 3900.2, min_step=0.4, breakeven_sl=3900.2) is True

    def test_a_small_move_that_does_not_reach_breakeven_still_needs_the_step(self):
        assert step_ok(False, 3900.5, 3900.3, min_step=0.4, breakeven_sl=3899.8) is False

    def test_once_past_breakeven_the_step_floor_applies_again(self):
        # stop already at/past break-even: a 0.05 improvement is just drift
        assert step_ok(False, 3899.0, 3898.95, min_step=0.4, breakeven_sl=3899.8) is False

    def test_breakeven_bypass_never_allows_a_loosening(self):
        assert step_ok(False, 3899.5, 3899.9, min_step=0.4, breakeven_sl=3899.8) is False
