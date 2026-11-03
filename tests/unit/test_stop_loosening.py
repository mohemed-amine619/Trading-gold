"""
tests/unit/test_stop_loosening.py — a stop may only move toward price, however
many stop paths run in one management pass, and each modification must be worth
a broker round trip.

Live-log evidence (SELL, ticket 2164109663): stop set to 3905.048, then one
second later "Modified SL -> 3905.221" (looser). Several paths run in sequence
on a position snapshot taken at the start of the pass; each compared against
that stale snapshot, so a later path could undo an earlier path's tightening.

The broker here is stateful: an order_send changes the stop immediately, while
the position snapshot the bot holds stays old, exactly as live.
"""

from __future__ import annotations

import time
import types

import pandas as pd
import pytest

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401

ENTRY, ATR = 3910.0, 5.0
INITIAL_SL, TP = 3919.0, 3890.0


class Broker:
    def __init__(self, direction):
        self.direction = direction                     # -1 sell, +1 buy
        self.sl = INITIAL_SL if direction == -1 else 2 * ENTRY - INITIAL_SL
        self.tp = TP if direction == -1 else 2 * ENTRY - TP
        self.sent = []
        self.fail = False

    def snapshot(self, price):
        move = (ENTRY - price) if self.direction == -1 else (price - ENTRY)
        return types.SimpleNamespace(
            ticket=555, symbol="XAUUSDm", type=1 if self.direction == -1 else 0,
            sl=self.sl, tp=self.tp, price_open=ENTRY, volume=0.01, magic=888992,
            profit=move * 1.0, swap=0.0, time=int(time.time()))

    def order_send(self, req):
        if self.fail:
            return types.SimpleNamespace(retcode=10013)
        self.sent.append(req["sl"])
        self.sl = req["sl"]
        return types.SimpleNamespace(retcode=10009)


def drive(run_module, monkeypatch, direction, prices, high_off=0.5, **cfg_overrides):
    """Run one management pass per price; return (broker, sl after each pass)."""
    broker = Broker(direction)
    state = {"price": prices[0]}
    mt5 = run_module.mt5
    monkeypatch.setattr(mt5, "positions_get", lambda **k: [broker.snapshot(state["price"])])
    monkeypatch.setattr(mt5, "order_send", broker.order_send)

    cfg = dict(run_module.CONFIG)
    cfg.update(dry_run=False, magic_number=888992, dynamic_sltp_enabled=False,
               manual_intel_enabled=False, structure_trailing=True, min_sl_step_atr=0.05,
               symbol="XAUUSDm", state_dir=None)
    cfg.update(cfg_overrides)
    bot = run_module.StandaloneAIBot(cfg)
    monkeypatch.setattr(bot, "_close_position", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the trade must not be closed in this scenario")))

    history = []
    for price in prices:
        state["price"] = price
        bid, ask = price - 0.1, price + 0.1
        df = pd.DataFrame([{"close": price, "atr": ATR, "high": price + high_off, "low": price - high_off}])
        tick = types.SimpleNamespace(bid=bid, ask=ask)
        bot._manage_positions(df, tick=tick, equity=500.0)
        history.append(broker.sl)
    return broker, history


def falling(n=240, start=3909.5, stop=3896.0):
    step = (start - stop) / n
    return [start - i * step for i in range(n)]


def rising(n=240, start=3910.5, stop=3924.0):
    step = (stop - start) / n
    return [start + i * step for i in range(n)]


class TestNeverLoosens:
    def test_a_sell_stop_never_moves_up(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        broker, history = drive(run_module, monkeypatch, -1, falling())
        assert broker.sent, "the scenario must exercise the stop paths"
        for earlier, later in zip(history, history[1:]):
            assert later <= earlier + 1e-9, f"stop loosened: {earlier} -> {later}"

    def test_a_buy_stop_never_moves_down(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        broker, history = drive(run_module, monkeypatch, +1, rising())
        assert broker.sent
        for earlier, later in zip(history, history[1:]):
            assert later >= earlier - 1e-9, f"stop loosened: {earlier} -> {later}"

    def test_every_sent_stop_is_tighter_than_the_previous_one(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        broker, _ = drive(run_module, monkeypatch, -1, falling())
        sent = [INITIAL_SL] + broker.sent
        assert all(b < a for a, b in zip(sent, sent[1:]))


class TestNoSpam:
    def test_modifications_are_worth_a_round_trip(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        broker, _ = drive(run_module, monkeypatch, -1, falling())
        breakeven = ENTRY - 0.15 * ATR - 1e-6          # entry less the breakeven buffer
        floor = 0.05 * ATR - 1e-6                      # min_sl_step_atr x ATR
        sent = [INITIAL_SL] + broker.sent
        small = [(a, b) for a, b in zip(sent, sent[1:])
                 if (a - b) < floor and not (a > breakeven >= b - 0.5)]
        assert small == [], f"modifications below the step floor: {small[:5]}"

    def test_far_fewer_modifications_than_passes(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        prices = falling(n=400)
        broker, _ = drive(run_module, monkeypatch, -1, prices)
        assert 0 < len(broker.sent) < len(prices) / 5

    def test_a_zero_step_floor_restores_the_old_chatter(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        prices = falling(n=400)
        quiet, _ = drive(run_module, monkeypatch, -1, prices)
        noisy, _ = drive(run_module, monkeypatch, -1, prices, min_sl_step_atr=0.0)
        assert len(noisy.sent) > len(quiet.sent)


class TestBrokerFailures:
    def test_a_rejected_modification_does_not_crash_or_loosen(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        broker = Broker(-1)
        broker.fail = True
        mt5 = run_module.mt5
        monkeypatch.setattr(mt5, "positions_get", lambda **k: [broker.snapshot(3900.0)])
        monkeypatch.setattr(mt5, "order_send", broker.order_send)
        cfg = dict(run_module.CONFIG)
        cfg.update(dry_run=False, magic_number=888992, dynamic_sltp_enabled=False,
                   manual_intel_enabled=False, symbol="XAUUSDm")
        bot = run_module.StandaloneAIBot(cfg)
        monkeypatch.setattr(bot, "_close_position", lambda *a, **k: None)
        df = pd.DataFrame([{"close": 3900.0, "atr": ATR, "high": 3900.5, "low": 3899.5}])
        bot._manage_positions(df, tick=types.SimpleNamespace(bid=3899.9, ask=3900.1), equity=500.0)
        assert broker.sl == INITIAL_SL                    # nothing was accepted


class TestModifySlGuard:
    """_modify_sl directly: the broker's live stop decides, not the stale snapshot."""

    def bot(self, run_module):
        cfg = dict(run_module.CONFIG)
        cfg.update(dry_run=False, magic_number=888992, symbol="XAUUSDm")
        return run_module.StandaloneAIBot(cfg)

    def setup_broker(self, run_module, monkeypatch, live_sl):
        sent = []
        mt5 = run_module.mt5
        monkeypatch.setattr(mt5, "positions_get",
                            lambda **k: [types.SimpleNamespace(sl=live_sl)] if live_sl is not None else [])
        monkeypatch.setattr(mt5, "order_send",
                            lambda req: (sent.append(req["sl"]) or types.SimpleNamespace(retcode=10009)))
        return sent

    def pos(self, sl, typ=1):
        return types.SimpleNamespace(ticket=1, symbol="XAUUSDm", type=typ, sl=sl, tp=3890.0,
                                     magic=888992)

    def test_a_stale_snapshot_cannot_loosen_a_sell(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        sent = self.setup_broker(run_module, monkeypatch, live_sl=3905.05)   # already tightened
        # snapshot still says 3905.29; 3905.22 "improves" on that but loosens the live stop
        assert self.bot(run_module)._modify_sl(self.pos(sl=3905.29), 3905.22) is False
        assert sent == []

    def test_a_real_improvement_is_sent(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        sent = self.setup_broker(run_module, monkeypatch, live_sl=3905.05)
        assert self.bot(run_module)._modify_sl(self.pos(sl=3905.29), 3904.90) is True
        assert sent == [3904.9]

    def test_a_stale_snapshot_cannot_loosen_a_buy(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        sent = self.setup_broker(run_module, monkeypatch, live_sl=3951.0)
        assert self.bot(run_module)._modify_sl(self.pos(sl=3950.0, typ=0), 3950.5) is False
        assert sent == []

    def test_if_the_broker_cannot_be_read_the_snapshot_is_used(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        sent = self.setup_broker(run_module, monkeypatch, live_sl=None)
        bot = self.bot(run_module)
        assert bot._modify_sl(self.pos(sl=3905.29), 3905.22) is True        # tighter than snapshot
        assert bot._modify_sl(self.pos(sl=3905.29), 3905.60) is False       # looser than snapshot
        assert sent == [3905.22]

    def test_no_stop_set_accepts_any_stop(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        sent = self.setup_broker(run_module, monkeypatch, live_sl=0.0)
        assert self.bot(run_module)._modify_sl(self.pos(sl=0.0), 3915.0) is True
        assert sent == [3915.0]

    def test_an_identical_stop_sends_nothing(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        sent = self.setup_broker(run_module, monkeypatch, live_sl=3905.05)
        assert self.bot(run_module)._modify_sl(self.pos(sl=3905.05), 3905.05) is True
        assert sent == []

    def test_a_refused_loosening_is_logged_once_not_every_tick(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        self.setup_broker(run_module, monkeypatch, live_sl=3905.05)
        bot = self.bot(run_module)
        with caplog.at_level("INFO"):
            for _ in range(20):
                bot._modify_sl(self.pos(sl=3905.29), 3905.22)
        assert sum("would loosen" in r.message for r in caplog.records) == 1


class TestTightenAfterKnob:
    def test_default_is_off_and_declared_in_the_schema(self, run_and_stub):
        run_module, _ = run_and_stub
        from core.config.schema import PARAMS
        assert run_module.CONFIG["min_profit_before_tighten_atr"] == 0.0
        spec = PARAMS["min_profit_before_tighten_atr"]
        assert spec.default == 0.0 and spec.env == "MIN_PROFIT_BEFORE_TIGHTEN_ATR"

    def test_structure_trailing_waits_for_the_configured_profit(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        # The trade is only 0.18 ATR in profit (0.9 of a 5.0 ATR). A swing high 3.0
        # above price gives the structure stop something to anchor to. With the
        # knob at 0.3 ATR it must wait; with it off it tightens at once.
        prices = [ENTRY - 1.0] * 3
        quiet = dict(enable_breakeven=False, early_protect_atr=9.0, trail_activate_atr=9.0)
        on, _ = drive(run_module, monkeypatch, -1, prices, high_off=3.0,
                      min_profit_before_tighten_atr=0.3, **quiet)
        off, _ = drive(run_module, monkeypatch, -1, prices, high_off=3.0,
                       min_profit_before_tighten_atr=0.0, **quiet)
        assert on.sent == [] and len(off.sent) >= 1
        assert all(sl < INITIAL_SL for sl in off.sent)
