"""
tests/unit/test_lock_distance.py — the stop distance of the progressive break-even /
profit-lock block is configurable (LOCK_MIN_DISTANCE_ATR).

It used to be hard-coded at 0.55 ATR for gold. The block only moves the stop to
break-even once price is far enough from it, so the EFFECTIVE break-even trigger
is that distance plus the break-even buffer (about 0.75 ATR), whatever
BREAKEVEN_TRIGGER_ATR says, and afterwards the stop trails 0.55 ATR behind price:
a normal pullback is a small win. That matches a +$1.41 average win.

Hand-worked case: SELL from 3910.0, ATR 5.0, break-even buffer 0.15 ATR (0.75),
so break-even is a stop at 3909.25. Price falls to 3906.0; the harness's ask is
3906.1, profit 3.9 (0.78 ATR).
  legacy 0.55 ATR (2.75): ask + 2.75 = 3908.85 <= 3909.25  -> the stop moves to 3908.85
  1.00 ATR (5.00):        ask + 5.00 = 3911.10 >  3909.25  -> NOT yet; the stop stays
"""

from __future__ import annotations

import pytest

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_stop_loosening import INITIAL_SL, drive

LOCK_ONLY = dict(structure_trailing=False, early_protect_atr=99.0, trail_activate_atr=99.0,
                 enable_breakeven=True, breakeven_trigger_atr=0.4, breakeven_buffer_atr=0.15,
                 profit_lock_ratio=0.5, min_profit_before_tighten_atr=0.0,
                 enable_partial_close=False)
PATH = [3909.0, 3907.0, 3906.0]


def final_sl(run_module, monkeypatch, **over):
    broker, history = drive(run_module, monkeypatch, -1, PATH, **{**LOCK_ONLY, **over})
    return history[-1]


class TestLockDistance:
    def test_default_is_the_legacy_behaviour(self, run_and_stub):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["lock_min_distance_atr"] == 0.0

    def test_legacy_moves_the_stop_to_break_even_at_about_0_75_atr(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        assert final_sl(run_module, monkeypatch) == pytest.approx(3908.85, abs=0.011)

    def test_an_explicit_055_is_identical_to_the_legacy_default(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        assert (final_sl(run_module, monkeypatch, lock_min_distance_atr=0.55)
                == final_sl(run_module, monkeypatch, lock_min_distance_atr=0.0))

    def test_a_wider_distance_delays_break_even(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        assert final_sl(run_module, monkeypatch, lock_min_distance_atr=1.0) == INITIAL_SL

    def test_the_stop_keeps_the_configured_distance_once_it_does_move(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        # Ask 3903.1, profit 6.9. Progressive lock: buffer 0.75 + (6.9 - 2.0) * 0.5 = 3.2 locked
        # -> 3906.8; the 1.0 ATR distance puts the stop at ask + 5.0 = 3908.1 (<= break-even 3909.25).
        broker, history = drive(run_module, monkeypatch, -1, [3909.0, 3906.0, 3903.0],
                                **{**LOCK_ONLY, "lock_min_distance_atr": 1.0})
        assert history[-1] == pytest.approx(3908.1, abs=0.011)

    def test_it_never_loosens_whatever_the_distance(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        path = [3909.0, 3906.0, 3903.0, 3908.0, 3902.0]            # a bounce in the middle
        for dist in (0.0, 0.55, 1.0, 2.0):
            # giveback off: this isolates how the STOP moves, not the bot's own giveback close
            broker, history = drive(run_module, monkeypatch, -1, path,
                                    **{**LOCK_ONLY, "lock_min_distance_atr": dist,
                                       "profit_lock_activate_atr": 99.0})
            assert all(b <= a + 1e-9 for a, b in zip(history, history[1:])), dist

    def test_the_setting_is_declared_with_the_right_bounds(self, run_and_stub):
        from core.config.schema import PARAMS
        spec = PARAMS["lock_min_distance_atr"]
        assert spec.default == 0.0 and spec.min == 0.0 and spec.env == "LOCK_MIN_DISTANCE_ATR"
