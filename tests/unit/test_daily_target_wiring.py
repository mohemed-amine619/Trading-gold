"""The daily-target state machine through the bot's real methods, plus the
config plane staying in sync and the NN sizing / entry rules."""

import types

import pytest

from core.config.schema import PARAMS
from core.risk.daily_state import DayState
from core.risk.nn_sizing import apply_nn_boost, nn_entry_ok, nn_size_factor
from core.settings import Settings
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot


def _pos(ticket, symbol="XAUUSDm", profit=0.0, magic=888999, buy=True, entry=2000.0, sl=0.0):
    return types.SimpleNamespace(ticket=ticket, symbol=symbol, profit=profit, swap=0.0,
                                 magic=magic, type=0 if buy else 1, price_open=entry,
                                 sl=sl, tp=0.0, volume=0.01)


@pytest.fixture
def bot(run_and_stub, tmp_path):                       # noqa: F811
    run_module, _ = run_and_stub
    b = make_bot(run_module, str(tmp_path / "r.json"),
                 daily_profit_target_money=70.0, profit_lock_buffer_usd=5.0,
                 daily_lock_action="close")
    b.closed, b.sent = [], []
    b._close_position = lambda pos, reason="": b.closed.append((pos.ticket, reason)) or True
    b.notifier = types.SimpleNamespace(send=lambda m: b.sent.append(m))
    b._realized = 0.0
    b._realized_today = lambda: b._realized
    return b


def test_below_target_nothing_happens_and_entries_stay_open(bot):
    bot._bot_positions_all = lambda: [_pos(1, profit=30.0)]
    bot._apply_daily_target()
    assert bot.closed == [] and bot._day_tracker.state is DayState.ACTIVE
    assert bot._entry_gate(400.0)[0] is True
    assert bot.status["day_state"] == "ACTIVE" and bot.status["day_net"] == 30.0


def test_floating_plus_realized_reaches_target_banks_it_and_halts(bot):
    bot._realized = 50.0
    bot._bot_positions_all = lambda: [_pos(1, profit=15.0), _pos(2, profit=8.0)]
    bot._apply_daily_target()
    assert sorted(t for t, _ in bot.closed) == [1, 2]
    assert all(r == "daily target" for _, r in bot.closed)
    allowed, mult, reason = bot._entry_gate(400.0)
    assert allowed is False and mult == 0.0 and "LOCKING" in reason
    assert len(bot.sent) == 1 and "LOCKING" in bot.sent[0]

    # Now flat with the gain banked: DAILY_TARGET_REACHED, latched.
    bot._realized = 73.0
    bot._bot_positions_all = lambda: []
    bot._apply_daily_target()
    assert bot._day_tracker.state is DayState.DAILY_TARGET_REACHED
    assert bot._entry_gate(400.0)[0] is False
    assert len(bot.sent) == 2 and "DAILY_TARGET_REACHED" in bot.sent[1]
    bot._apply_daily_target()                          # no duplicate alert
    assert len(bot.sent) == 2


def test_only_the_bots_magic_number_counts_toward_the_target(run_and_stub, bot):
    run_module, _ = run_and_stub
    mixed = [_pos(1, profit=10.0), _pos(2, profit=500.0, magic=0)]     # manual trade: big
    run_module.mt5.positions_get = lambda *a, **k: mixed
    assert [p.ticket for p in bot._bot_positions_all()] == [1]
    assert bot._floating_bot_pnl() == 10.0
    bot._apply_daily_target()
    assert bot.closed == [] and bot._day_tracker.state is DayState.ACTIVE


def test_positions_on_other_symbols_are_not_closed_by_this_bot(run_and_stub, bot):
    bot._realized = 80.0
    bot._bot_positions_all = lambda: [_pos(1), _pos(2, symbol="EURUSDm")]
    bot._apply_daily_target()
    assert [t for t, _ in bot.closed] == [1]


def test_ratchet_mode_only_proposes_a_tighter_stop(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    b = make_bot(run_module, str(tmp_path / "r2.json"), daily_profit_target_money=70.0,
                 profit_lock_buffer_usd=5.0, daily_lock_action="ratchet")
    b.notifier = types.SimpleNamespace(send=lambda m: None)
    b._realized_today = lambda: 60.0
    b._close_position = lambda *a, **k: pytest.fail("ratchet must not close here")
    calls = []
    b._modify_sl = lambda pos, sl: calls.append((pos.ticket, sl)) or True
    b._bot_positions_all = lambda: [_pos(1, profit=12.0, entry=2000.0)]
    tick = types.SimpleNamespace(bid=2010.0, ask=2010.2)
    run_module.mt5.symbol_info_tick = lambda s: tick
    b._apply_daily_target()
    assert calls == [(1, pytest.approx(2008.0))]       # keeps 80% of the 10-point gain
    assert calls[0][1] > 2000.0                         # above entry: tightens a loss-free stop


def test_zero_target_disables_everything(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    b = make_bot(run_module, str(tmp_path / "r3.json"), daily_profit_target_money=0.0)
    b._realized_today = lambda: 1000.0
    b._bot_positions_all = lambda: [_pos(1, profit=500.0)]
    b._close_position = lambda *a, **k: pytest.fail("disabled target must not close")
    b._apply_daily_target()
    assert b._entry_gate(400.0)[0] is True


# ── config plane ────────────────────────────────────────────────────────────

def test_new_defaults_match_across_schema_run_and_settings(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    for var in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH",
                "DAILY_PROFIT_TARGET_MONEY", "PROFIT_LOCK_BUFFER_USD", "DAILY_LOCK_ACTION",
                "NN_MIN_CONFIDENCE"):
        monkeypatch.delenv(var, raising=False)       # the fixture blanks some of these
    s = Settings(_env_file=None)
    for key, want in (("daily_profit_target_money", 70.0), ("profit_lock_buffer_usd", 5.0),
                      ("daily_lock_action", "close"), ("nn_min_confidence", 0.72)):
        assert PARAMS[key].default == want, key
        assert run_module.CONFIG[key] == want, key
        assert getattr(s, key) == want, key
        assert s.as_legacy_config()[key] == want, key


def test_new_keys_are_declared_so_the_dashboard_renders_them():
    for key in ("daily_profit_target_money", "profit_lock_buffer_usd", "daily_lock_action",
                "nn_min_confidence", "nn_entry_required", "nn_size_boost_max"):
        assert key in PARAMS and PARAMS[key].env


# ── NN sizing / entry rule ──────────────────────────────────────────────────

def test_nn_boost_scales_with_confidence_above_threshold_and_is_capped():
    f = lambda c: nn_size_factor(1, 1, c, 0.72, 0.5)           # noqa: E731
    assert f(0.72) == 1.0 and f(0.60) == 1.0
    assert 1.0 < f(0.80) < f(0.90) < f(1.0) == 1.5
    assert nn_size_factor(-1, 1, 0.95, 0.72, 0.5) == 1.0      # network disagrees
    assert nn_size_factor(0, 1, 0.95, 0.72, 0.5) == 1.0       # network neutral
    assert nn_size_factor(1, 1, 0.95, 0.72, 0.0) == 1.0       # disabled


def test_nn_boost_never_applies_to_an_already_reduced_size():
    assert apply_nn_boost(0.5, 1.5) == 0.5                      # drawdown ladder rung
    assert apply_nn_boost(0.7, 1.5) == 0.7                      # losing streak
    assert apply_nn_boost(1.0, 1.5) == 1.5


def test_sizing_has_no_profit_target_input():
    import inspect
    params = inspect.signature(nn_size_factor).parameters
    assert not any("profit" in p or "target" in p or "remaining" in p for p in params)


def test_nn_entry_requirement():
    assert nn_entry_ok(1, 1, 0.80, 0.72)
    assert not nn_entry_ok(1, 1, 0.70, 0.72)
    assert not nn_entry_ok(-1, 1, 0.99, 0.72)
    assert not nn_entry_ok(0, 1, 0.99, 0.72)
    assert not nn_entry_ok(1, 0, 0.99, 0.72)
