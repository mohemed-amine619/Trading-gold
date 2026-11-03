"""
tests/unit/test_daily_dd_off.py — MAX_DAILY_DD=0 means "no daily stop".

Seen live: with MAX_DAILY_DD=0 the bot logged "Daily DD limit hit (0.0%) - no new trades
today" at startup, because 0.0 >= 0.0. With equity_protect_close_all it would also have
closed every position. The weekly/total caps and the soft ladder already treated 0 as off.
"""

from __future__ import annotations

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot


def bot_with(run_module, tmp_path, **cfg):
    bot = make_bot(run_module, str(tmp_path / "r.json"), **cfg)
    bot.daily_start = 400.0
    return bot


def test_a_zero_cap_never_halts_even_at_zero_drawdown(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = bot_with(run_module, tmp_path, max_daily_dd=0.0, max_weekly_dd=0.0, max_total_dd=0.0)
    assert bot._daily_dd_ok(400.0) is True


def test_a_zero_cap_does_not_close_positions_or_halt_on_a_big_loss(run_and_stub, tmp_path, monkeypatch):
    run_module, _ = run_and_stub
    bot = bot_with(run_module, tmp_path, max_daily_dd=0.0, max_weekly_dd=0.0, max_total_dd=0.0,
                   equity_protect_close_all=True)
    closed = []
    monkeypatch.setattr(bot, "_close_all", lambda reason: closed.append(reason))
    assert bot._daily_dd_ok(300.0) is True              # -25% on the day
    assert closed == [] and bot._entry_gate(300.0)[0] is True


def test_a_real_cap_still_fires(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = bot_with(run_module, tmp_path, max_daily_dd=0.03)
    assert bot._daily_dd_ok(399.0) is True
    assert bot._daily_dd_ok(387.0) is False             # -3.25%
