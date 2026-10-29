"""
tests/unit/test_daily_target.py — daily profit lock.

The lock must:
* stay off unless a positive target is configured (default behaviour is
  unchanged),
* trip only on a real gain over the day's starting equity,
* never trip on a missing or non-positive baseline,
* be declared in the config schema with a bounded, disabled-by-default value.

Imports only the pure module and the schema, never run.py: run.py imports
MetaTrader5 at module level and cannot be loaded on Linux.
"""

from __future__ import annotations

import pytest

from core.config.schema import PARAMS
from core.risk.daily_target import daily_gain, profit_lock_reached


class TestDailyGain:
    def test_gain_is_fraction_of_start(self):
        assert daily_gain(1000.0, 1050.0) == pytest.approx(0.05)

    def test_loss_is_negative(self):
        assert daily_gain(1000.0, 950.0) == pytest.approx(-0.05)

    @pytest.mark.parametrize("start", [0.0, -10.0, None])
    def test_missing_baseline_reads_as_zero_not_profit(self, start):
        assert daily_gain(start, 500.0) == 0.0


class TestProfitLockReached:
    def test_disabled_when_target_is_zero(self):
        assert profit_lock_reached(1000.0, 5000.0, 0.0) is False

    def test_disabled_when_target_is_none(self):
        assert profit_lock_reached(1000.0, 5000.0, None) is False

    def test_not_reached_below_target(self):
        assert profit_lock_reached(1000.0, 1019.0, 0.02) is False

    def test_reached_exactly_at_target(self):
        assert profit_lock_reached(1000.0, 1020.0, 0.02) is True

    def test_reached_above_target(self):
        assert profit_lock_reached(1000.0, 1100.0, 0.02) is True

    def test_loss_never_trips_the_lock(self):
        assert profit_lock_reached(1000.0, 900.0, 0.02) is False

    def test_unknown_baseline_never_trips_the_lock(self):
        assert profit_lock_reached(0.0, 5000.0, 0.02) is False


class TestSchemaDeclaration:
    def test_parameter_is_declared(self):
        assert "daily_profit_target" in PARAMS

    def test_disabled_by_default(self):
        spec = PARAMS["daily_profit_target"]
        assert spec.default == 0.0

    def test_bounds_cap_the_target(self):
        spec = PARAMS["daily_profit_target"]
        assert spec.min == 0.0
        assert spec.max == 0.50
        assert spec.percent is True
        assert spec.env == "DAILY_PROFIT_TARGET"
