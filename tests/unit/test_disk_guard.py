"""
tests/unit/test_disk_guard.py — a nearly-full data disk stops NEW entries.

The Windows box ran out of space while live: log writes failed, and so could
the risk-state and trade-history writes. The guard is pure (usage is injected)
and is wired into run.py's ``_entry_gate``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from core.risk.disk_guard import free_mb, low_disk_reason
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot

MB = 1024 * 1024


def usage_with(free_mb_value):
    return lambda _path: SimpleNamespace(free=int(free_mb_value * MB))


def broken_usage(_path):
    raise OSError("cannot stat")


class TestPureGuard:
    def test_blocks_below_the_floor(self):
        reason, free = low_disk_reason("x", 500, usage_with(120))
        assert "low disk space: 120 MB free < 500 MB" in reason
        assert round(free) == 120

    def test_allows_at_and_above_the_floor(self):
        assert low_disk_reason("x", 500, usage_with(500))[0] == ""
        assert low_disk_reason("x", 500, usage_with(50_000))[0] == ""

    def test_zero_floor_disables_the_guard(self):
        assert low_disk_reason("x", 0, usage_with(0)) == ("", None)

    def test_unmeasurable_is_unknown_not_zero_and_does_not_block(self):
        assert free_mb("x", broken_usage) is None
        assert low_disk_reason("x", 500, broken_usage) == ("", None)

    def test_a_real_path_is_measurable(self, tmp_path):
        assert free_mb(str(tmp_path)) is not None


class TestRunWiring:
    def _bot(self, run_module, tmp_path, monkeypatch, free, **cfg):
        monkeypatch.setattr(run_module, "_low_disk_reason",
                            lambda path, floor: low_disk_reason(path, floor, usage_with(free)))
        bot = make_bot(run_module, str(tmp_path / "r.json"), **cfg)
        bot.daily_start = 389.0
        return bot

    def test_default_floor_is_500_mb(self, run_and_stub):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["min_free_disk_mb"] == 500.0

    def test_the_real_gate_refuses_a_new_entry_on_a_full_disk(
            self, run_and_stub, tmp_path, monkeypatch):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, monkeypatch, free=50, min_free_disk_mb=500)
        allowed, mult, reason = bot._entry_gate(389.0)
        assert (allowed, mult) == (False, 0.0) and "low disk space" in reason

    def test_plenty_of_space_does_not_block(self, run_and_stub, tmp_path, monkeypatch):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, monkeypatch, free=50_000, min_free_disk_mb=500)
        assert bot._low_disk_reason() == ""

    def test_trip_and_recovery_are_each_logged_once(
            self, run_and_stub, tmp_path, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, monkeypatch, free=50, min_free_disk_mb=500)
        with caplog.at_level(logging.INFO):
            bot._low_disk_reason()
            bot._low_disk_reason()
            monkeypatch.setattr(run_module, "_low_disk_reason",
                                lambda path, floor: low_disk_reason(path, floor, usage_with(9_000)))
            bot._low_disk_reason()
            bot._low_disk_reason()
        msgs = [r.getMessage() for r in caplog.records]
        assert sum("no new entries until space is freed" in m for m in msgs) == 1
        assert sum("disk space recovered" in m for m in msgs) == 1

    def test_the_setting_is_declared_for_the_dashboard(self):
        from core.config.schema import PARAMS
        assert PARAMS["min_free_disk_mb"].env == "MIN_FREE_DISK_MB"
