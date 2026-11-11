"""
Early protection used to cut winners to scratches (live: SELL 4017.80, TP 4005.61,
stop moved to 4017.16 after a 0.35-ATR move, closed +0.64). It now defaults to
0.8 ATR and never fires before the break-even trigger.

Harness (see test_lock_distance.py): SELL from 3910.0, ATR 5.0, the ask is price
+ 0.1. Price 3908.2 -> ask 3908.3 -> profit 1.7 = 0.34 ATR.
"""

from __future__ import annotations

import pytest

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_stop_loosening import INITIAL_SL, drive

# Only early protection can move the stop: the lock block's distance is huge,
# trailing / giveback / partials / structure are off.
EARLY_ONLY = dict(structure_trailing=False, trail_activate_atr=99.0, profit_lock_activate_atr=99.0,
                  enable_partial_close=False, lock_min_distance_atr=99.0,
                  min_profit_before_tighten_atr=0.0, breakeven_buffer_atr=0.15)
SMALL_GAIN = [3909.0, 3908.2]          # 0.34 ATR in profit
BIG_GAIN = [3909.0, 3905.5]            # 0.88 ATR in profit


def final_sl(run_module, monkeypatch, path, **over):
    _, history = drive(run_module, monkeypatch, -1, path, **{**EARLY_ONLY, **over})
    return history[-1]


def test_new_default_is_08(run_and_stub):
    run_module, _ = run_and_stub
    assert run_module.CONFIG["early_protect_atr"] == 0.8
    from core.config.schema import PARAMS
    assert PARAMS["early_protect_atr"].default == 0.8


def test_the_old_03_still_clips_when_breakeven_is_off(run_and_stub, monkeypatch):
    """Reproduces the live behaviour: 40% of a 0.34-ATR gain locked."""
    run_module, _ = run_and_stub
    sl = final_sl(run_module, monkeypatch, SMALL_GAIN, early_protect_atr=0.3,
                  enable_breakeven=False)
    assert sl == pytest.approx(3910.0 - 1.7 * 0.4, abs=0.011)


def test_never_before_the_breakeven_trigger(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    sl = final_sl(run_module, monkeypatch, SMALL_GAIN, early_protect_atr=0.3,
                  enable_breakeven=True, breakeven_trigger_atr=0.8)
    assert sl == INITIAL_SL                                   # untouched at 0.34 ATR


def test_default_leaves_a_small_winner_alone(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    sl = final_sl(run_module, monkeypatch, SMALL_GAIN, enable_breakeven=True,
                  breakeven_trigger_atr=0.4)
    assert sl == INITIAL_SL


def test_it_still_protects_once_the_trade_has_room(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    sl = final_sl(run_module, monkeypatch, BIG_GAIN, enable_breakeven=True,
                  breakeven_trigger_atr=0.8)
    assert sl < INITIAL_SL and sl == pytest.approx(3910.0 - 4.4 * 0.4, abs=0.011)


def test_trigger_helper(run_and_stub):
    run_module, _ = run_and_stub
    f = run_module._early_protect_trigger
    assert f({"early_protect_atr": 0.3, "breakeven_trigger_atr": 0.8}) == 0.8
    assert f({"early_protect_atr": 1.2, "breakeven_trigger_atr": 0.8}) == 1.2
    assert f({"early_protect_atr": 0.3, "breakeven_trigger_atr": 0.8,
              "enable_breakeven": False}) == 0.3


def test_replay_uses_the_same_rule():
    """The offline replay must not lock profit below the break-even trigger either.
    A SELL runs 0.4 ATR in profit, pulls back, then reaches its target: with the
    old rule the pullback stops it out for a scratch, with the new rule it wins."""
    import pandas as pd
    from core.backtesting.management_replay import ManagementParams, ReplayTrade, simulate_trade
    idx = pd.date_range("2026-11-10 17:00", periods=4, freq="5min")
    bars = pd.DataFrame({"open": [4018.0, 4016.0, 4017.6, 4012.0],
                         "high": [4018.2, 4017.0, 4018.0, 4012.5],
                         "low": [4016.0, 4016.0, 4012.0, 4005.0],
                         "close": [4016.0, 4016.9, 4012.5, 4005.5]}, index=idx)
    trade = ReplayTrade("t", -1, idx[0], 4018.0, 4026.3, 4005.6, 0.01, 4.6)
    common = dict(trail_activate_atr=99.0, profit_lock_activate_atr=99.0,
                  lock_min_distance_atr=99.0, breakeven_trigger_atr=0.8)
    old = simulate_trade(trade, bars, ManagementParams(early_protect_atr=0.3,
                                                       enable_breakeven=False, **common))
    new = simulate_trade(trade, bars, ManagementParams(early_protect_atr=0.3, **common))
    assert old.reason == "sl" and old.move < 1.5          # clipped to a scratch
    assert new.reason == "tp" and new.move > 12.0          # ran to the target


def test_stop_changes_say_which_rule_moved_them(run_and_stub, monkeypatch, caplog):
    run_module, _ = run_and_stub
    import logging
    with caplog.at_level(logging.INFO):
        final_sl(run_module, monkeypatch, BIG_GAIN, enable_breakeven=True,
                 breakeven_trigger_atr=0.8)
    lines = [r.getMessage() for r in caplog.records if "Modify SL" in r.getMessage()
             or "Modified SL" in r.getMessage()]
    assert lines and all("(early protect)" in ln for ln in lines)


def test_manual_trade_protection_reads_the_environment():
    src = open("run.py", encoding="utf-8").read()
    assert 'os.getenv("AI_PROTECT_MANUAL_TRADES"' in src
    assert 'os.getenv("MANUAL_TRADE_SL_ATR"' in src


def test_critic_label_is_honest(run_and_stub):
    run_module, _ = run_and_stub
    assert "not an LLM" in run_module._critic_label({"llm_provider": "none"})
    assert "not an LLM" in run_module._critic_label({})
    assert run_module._critic_label({"llm_provider": "openai_compatible"}) == "LLM: openai_compatible"
