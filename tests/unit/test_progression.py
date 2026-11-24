"""
tests/unit/test_progression.py — martingale, anti-martingale, Kelly, grid and DCA
(core/risk/progression.py). All OFF by default; these tests pin the caps that keep
them bounded.
"""

from __future__ import annotations

import pytest

from core.risk.progression import (
    anti_martingale_multiplier, grid_add, kelly_fraction, kelly_multiplier,
    martingale_multiplier, progression_multiplier, total_risk_if_stopped,
)


class TestMartingale:
    def test_doubles_per_consecutive_loss(self):
        assert martingale_multiplier([5, -1]) == 2.0
        assert martingale_multiplier([5, -1, -1]) == 4.0

    def test_capped_at_max_steps(self):
        assert martingale_multiplier([-1] * 10, 2.0, 3) == 8.0

    def test_a_win_or_breakeven_resets(self):
        assert martingale_multiplier([-1, -1, 3]) == 1.0
        assert martingale_multiplier([-1, -1, 0.0]) == 1.0

    def test_factor_below_one_is_refused(self):
        with pytest.raises(ValueError):
            martingale_multiplier([-1], 0.5)


class TestAntiMartingale:
    def test_presses_winners_and_resets_on_loss(self):
        assert anti_martingale_multiplier([1, 1], 1.5, 2) == pytest.approx(2.25)
        assert anti_martingale_multiplier([1, 1, 1, 1], 1.5, 2) == pytest.approx(2.25)
        assert anti_martingale_multiplier([1, 1, -1]) == 1.0


class TestKelly:
    def test_fraction_by_hand(self):
        # 60% winners, average win 2, average loss 1 -> W - (1-W)/R = 0.6 - 0.4/2 = 0.4
        pnls = [2, 2, 2, -1, -1] * 10
        assert kelly_fraction(pnls) == pytest.approx(0.4)

    def test_multiplier_scales_risk_to_a_fraction_of_kelly(self):
        pnls = [2, 2, 2, -1, -1] * 10
        # quarter Kelly = 0.10 of equity; base risk 1% -> x10, capped at 2
        assert kelly_multiplier(pnls, 0.01, 0.25, 30, cap=2.0) == 2.0
        assert kelly_multiplier(pnls, 0.08, 0.25, 30, cap=2.0) == pytest.approx(1.25)

    def test_no_edge_shrinks_never_grows(self):
        assert kelly_multiplier([1, -2] * 20, 0.01) == 0.25

    def test_too_few_trades_leaves_size_alone(self):
        assert kelly_multiplier([2, -1] * 5, 0.01, min_trades=30) == 1.0


class TestProgressionMode:
    def test_off_and_unknown_modes_change_nothing(self):
        assert progression_multiplier("off", [-1, -1])[0] == 1.0
        assert progression_multiplier("typo", [-1, -1])[0] == 1.0

    def test_the_absolute_cap_wins_over_the_steps(self):
        m, why = progression_multiplier("martingale", [-1] * 6, factor=2.0, max_steps=5, max_mult=4.0)
        assert m == 4.0 and "martingale" in why


class TestGrid:
    kw = dict(first_entry=2000.0, first_sl=1990.0, first_tp=2020.0, atr=2.0)

    def test_first_add_needs_one_spacing_against_the_entry(self):
        assert not grid_add(1, price=1999.0, n_open=1, **self.kw).allowed
        plan = grid_add(1, price=1998.0, n_open=1, **self.kw)
        assert plan.allowed and plan.level == 1 and plan.size_mult == 1.0

    def test_adds_share_the_first_stop_and_target(self):
        plan = grid_add(1, price=1997.5, n_open=1, **self.kw)
        assert plan.stop == 1990.0 and plan.target == 2020.0

    def test_second_add_needs_two_spacings(self):
        assert not grid_add(1, price=1997.0, n_open=2, **self.kw).allowed
        assert grid_add(1, price=1996.0, n_open=2, **self.kw).allowed

    def test_max_levels(self):
        assert not grid_add(1, price=1992.0, n_open=4, max_levels=3, **self.kw).allowed

    def test_never_adds_without_a_stop_or_through_it(self):
        assert not grid_add(1, 2000.0, 0.0, 2020.0, 1998.0, 2.0, 1).allowed
        assert not grid_add(1, price=1989.0, n_open=1, **self.kw).allowed

    def test_short_side_mirrors(self):
        plan = grid_add(-1, 2000.0, 2010.0, 1980.0, 2002.5, 2.0, 1)
        assert plan.allowed and plan.stop == 2010.0

    def test_dca_scales_each_add(self):
        p1 = grid_add(1, price=1998.0, n_open=1, mode="dca", dca_scale=1.5, **self.kw)
        p2 = grid_add(1, price=1996.0, n_open=2, mode="dca", dca_scale=1.5, **self.kw)
        assert p1.size_mult == 1.5 and p2.size_mult == pytest.approx(2.25)

    def test_off_by_default(self):
        assert not grid_add(1, price=1990.5, n_open=1, mode="off", **self.kw).allowed

    def test_total_risk_is_known_before_the_add(self):
        legs = [(2000.0, 0.1), (1998.0, 0.1), (1996.0, 0.1)]
        # (10 + 8 + 6) points x 0.1 lot x 100 per point per lot
        assert total_risk_if_stopped(1, 1990.0, legs, 100.0) == pytest.approx(240.0)
