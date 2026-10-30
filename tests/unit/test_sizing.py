"""
tests/unit/test_sizing.py — position size that cannot silently exceed the
risk budget (plan criterion A2).

Hand-worked contract used throughout: a 10.0 stop on tick_size 0.01 /
tick_value 1.0 is 1000 ticks, so one lot loses 1000.0 when stopped out.
Therefore a budget of 500 buys 0.5 lot, and the smallest lot (0.01) risks 10.0.

The pure function is tested directly. ``_size_lot`` in run.py is covered
through the stubbed-MT5 fixture the other run.py tests already use.
"""

from __future__ import annotations

import math

import pytest

from core.risk.sizing import size_within_budget
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401

SL = 10.0
TICK = dict(tick_size=0.01, tick_value=1.0)
LOTS = dict(vol_min=0.01, vol_max=10.0, vol_step=0.01)


def size(budget, **overrides):
    kwargs = {**TICK, **LOTS, **overrides}
    return size_within_budget(budget, SL, **kwargs)


class TestVolume:
    def test_budget_buys_the_expected_lot(self):
        r = size(500.0)
        assert r.volume == pytest.approx(0.5)
        assert r.risk_amount == pytest.approx(500.0, rel=1e-6)
        assert r.forced_min_lot is False

    def test_volume_floors_to_the_step_and_never_exceeds_budget(self):
        r = size(555.0)                       # 0.555 lot -> 0.55
        assert r.volume == pytest.approx(0.55)
        assert r.risk_amount <= 555.0

    def test_volume_is_clamped_to_the_broker_maximum(self):
        r = size(50_000.0)                    # 50 lots wanted, 10 allowed
        assert r.volume == 10.0

    @pytest.mark.parametrize("budget", [37.0, 123.0, 500.0, 999.0, 4200.0, 77777.0])
    def test_matches_the_legacy_formula_when_the_budget_covers_a_lot(self, budget):
        legacy = max(0.01, min(math.floor(budget / (SL / (0.01 + 1e-10) * 1.0) / 0.01) * 0.01, 10.0))
        assert size(budget).volume == pytest.approx(legacy)


class TestMinimumLotOvershoot:
    def test_minimum_lot_is_reported_as_forced_with_its_real_risk(self):
        r = size(4.0)                         # budget buys 0.004 lot
        assert r.volume == pytest.approx(0.01)
        assert r.forced_min_lot is True
        assert r.risk_amount == pytest.approx(10.0, rel=1e-6)   # 2.5x the budget

    def test_overshoot_limit_zero_keeps_the_legacy_behaviour(self):
        assert size(4.0, max_overshoot=0.0).volume == pytest.approx(0.01)

    def test_refused_when_minimum_lot_exceeds_the_limit(self):
        r = size(4.0, max_overshoot=2.0)      # 10.0 > 4.0 * 2
        assert r.volume is None
        assert "minimum lot" in r.reason
        assert r.risk_amount == pytest.approx(10.0, rel=1e-6)

    def test_allowed_when_minimum_lot_is_within_the_limit(self):
        assert size(4.0, max_overshoot=3.0).volume == pytest.approx(0.01)

    def test_limit_does_not_affect_a_trade_the_budget_can_afford(self):
        assert size(500.0, max_overshoot=1.5).volume == pytest.approx(0.5)


class TestCannotSize:
    @pytest.mark.parametrize("field", [
        "tick_size", "tick_value", "vol_min", "vol_max", "vol_step"])
    @pytest.mark.parametrize("bad", [None, 0.0, -1.0])
    def test_unusable_broker_data_is_never_the_minimum_lot(self, field, bad):
        r = size(500.0, **{field: bad})
        assert r.volume is None
        assert r.reason.startswith("cannot size")

    @pytest.mark.parametrize("sl", [0.0, -5.0, None])
    def test_unusable_stop_distance(self, sl):
        r = size_within_budget(500.0, sl, **TICK, **LOTS)
        assert r.volume is None

    @pytest.mark.parametrize("budget", [0.0, -10.0, None])
    def test_no_budget_means_no_trade(self, budget):
        assert size(budget).volume is None

    def test_inverted_volume_limits(self):
        assert size(500.0, vol_min=1.0, vol_max=0.5).volume is None


class TestRunPyWiring:
    def _cfg(self, run_module, **overrides):
        cfg = dict(run_module.CONFIG)
        cfg.update({"risk_per_trade": 0.01, "dynamic_risk": False, **overrides})
        return cfg

    def _call(self, run_module, cfg, equity=400.0, tick_value=1.0):
        return run_module._size_lot(
            account_equity=equity, sl_dist_price=SL, tick_size=0.01,
            tick_value=tick_value, vol_min=0.01, vol_max=10.0, vol_step=0.01,
            cfg=cfg, adx=20.0, ai_conf=0.0)

    def test_overshoot_is_off_by_default(self, run_and_stub):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["max_risk_overshoot"] == 0.0

    def test_missing_tick_value_refuses_instead_of_minimum_lot(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self._call(run_module, self._cfg(run_module), tick_value=0.0) is None

    def test_default_config_still_allows_the_minimum_lot(self, run_and_stub):
        run_module, _ = run_and_stub
        # equity 400 x 1% = 4.0 budget; the minimum lot risks 10.0.
        vol = self._call(run_module, self._cfg(run_module))
        assert vol == pytest.approx(0.01)

    def test_overshoot_limit_refuses_the_oversized_minimum_lot(self, run_and_stub):
        run_module, _ = run_and_stub
        cfg = self._cfg(run_module, max_risk_overshoot=2.0)
        assert self._call(run_module, cfg) is None

    def test_overshoot_limit_leaves_affordable_trades_alone(self, run_and_stub):
        run_module, _ = run_and_stub
        cfg = self._cfg(run_module, max_risk_overshoot=2.0)
        assert self._call(run_module, cfg, equity=50_000.0) == pytest.approx(0.5)
