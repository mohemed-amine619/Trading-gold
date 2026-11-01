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


from core.risk.daily_target import money_target_reached  # noqa: E402


class TestMoneyTarget:
    def test_disabled_when_zero_or_none(self):
        assert money_target_reached(389.0, 1000.0, 0.0) is False
        assert money_target_reached(389.0, 1000.0, None) is False

    def test_not_reached_below_target(self):
        assert money_target_reached(389.0, 458.0, 70.0) is False   # +69

    def test_reached_at_and_above_target(self):
        assert money_target_reached(389.0, 459.0, 70.0) is True    # +70
        assert money_target_reached(389.0, 500.0, 70.0) is True

    def test_loss_never_trips_it(self):
        assert money_target_reached(389.0, 300.0, 70.0) is False

    @pytest.mark.parametrize("start", [0.0, -1.0, None])
    def test_unknown_baseline_never_trips_it(self, start):
        assert money_target_reached(start, 5000.0, 70.0) is False

    def test_declared_in_the_schema_and_off_by_default(self):
        spec = PARAMS["daily_profit_target_money"]
        assert spec.default == 0.0 and spec.min == 0.0
        assert spec.env == "DAILY_PROFIT_TARGET_MONEY"


from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: E402,F401
from tests.unit.test_risk_ladder import make_bot  # noqa: E402


class TestEntryGateWiring:
    """The lock through the bot's real _entry_gate, not just the pure function."""

    def _bot(self, run_module, tmp_path, **cfg):
        bot = make_bot(run_module, str(tmp_path / "r.json"), **cfg)
        bot.daily_start = 389.0
        return bot

    def test_money_target_blocks_new_entries_at_plus_70(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, daily_profit_target_money=70.0)
        allowed, mult, reason = bot._entry_gate(459.0)
        assert allowed is False and mult == 0.0
        assert "profit target" in reason

    def test_money_target_allows_entries_just_below_it(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, daily_profit_target_money=70.0)
        assert bot._entry_gate(458.0)[0] is True

    def test_percent_target_blocks_new_entries(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, daily_profit_target=0.10)
        assert bot._entry_gate(389.0 * 1.10)[0] is False
        assert bot._entry_gate(389.0 * 1.09)[0] is True

    def test_either_target_trips_the_lock(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, daily_profit_target=0.50,
                        daily_profit_target_money=70.0)
        assert bot._entry_gate(459.0)[0] is False        # money first, 18% < 50%

    def test_default_config_never_locks(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["daily_profit_target"] == 0.0
        assert run_module.CONFIG["daily_profit_target_money"] == 0.0
        bot = self._bot(run_module, tmp_path)
        assert bot._entry_gate(1000.0)[0] is True        # +157%, still allowed

    def test_a_loss_is_never_a_profit_lock(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, daily_profit_target_money=70.0)
        assert bot._entry_gate(380.0)[0] is True

    def test_the_lock_only_gates_entries_not_position_management(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, daily_profit_target_money=70.0,
                        max_daily_dd=0.03)
        # _daily_dd_ok owns the hard rungs and returns True on a winning day, so
        # step() still reaches _manage_positions while entries are locked.
        assert bot._daily_dd_ok(459.0) is True
