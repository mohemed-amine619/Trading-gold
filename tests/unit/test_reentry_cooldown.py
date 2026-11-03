"""
tests/unit/test_reentry_cooldown.py — a pause on one direction after a losing trade.

Real case: 45 entries within 15 minutes of the previous one in the same
direction, and a 12-trade losing streak. The pause is per direction, counted in
closed bars, and off by default.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from core.config.schema import PARAMS
from core.risk.cooldown import DirectionCooldown
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_trade_end_log import ATR, BOT_MAGIC, Rig, deal, position


class TestPureCooldown:
    def test_blocks_exactly_the_next_n_bars(self):
        cd = DirectionCooldown()
        cd.record_loss(-1, bar=10, bars=3)               # a SELL lost while bar 10 was current
        assert [cd.blocked(-1, b) for b in (10, 11, 12, 13, 14)] == [True, True, True, True, False]
        # bars 11, 12, 13 are the three blocked ones; 10 is the bar it ended on
        assert [cd.remaining(-1, b) for b in (11, 12, 13, 14)] == [3, 2, 1, 0]

    def test_the_other_direction_is_not_delayed(self):
        cd = DirectionCooldown()
        cd.record_loss(-1, 10, 5)
        assert cd.blocked(1, 11) is False and cd.remaining(1, 11) == 0

    def test_off_when_bars_is_zero_or_negative(self):
        cd = DirectionCooldown()
        cd.record_loss(1, 10, 0)
        cd.record_loss(1, 10, -3)
        assert cd.blocked(1, 11) is False

    def test_a_second_loss_extends_never_shortens(self):
        cd = DirectionCooldown()
        cd.record_loss(1, 10, 6)                         # until 16
        cd.record_loss(1, 11, 2)                         # would be 13: must not shorten
        assert cd.remaining(1, 12) == 5
        cd.record_loss(1, 15, 6)                         # until 21: extends
        assert cd.remaining(1, 16) == 6

    def test_an_unknown_direction_is_ignored(self):
        cd = DirectionCooldown()
        cd.record_loss(0, 1, 5)
        assert cd.blocked(0, 2) is False

    def test_a_direction_that_never_lost_is_never_blocked_whatever_the_bar(self):
        cd = DirectionCooldown()
        for bar in (0, 1, 50):
            assert cd.remaining(1, bar) == 0 and cd.blocked(-1, bar) is False

    def test_clear(self):
        cd = DirectionCooldown()
        cd.record_loss(1, 1, 5)
        cd.clear()
        assert cd.blocked(1, 2) is False


class TestRecordedFromTradeEnds:
    def end_trade(self, run_module, monkeypatch, net, magic=BOT_MAGIC, direction=-1, deals=True, **cfg):
        rig = Rig(run_module, monkeypatch, reentry_cooldown_bars=3, ai_protect_manual_trades=False,
                  smart_manage_manual_trades=False, **cfg)
        if deals:
            rig.deals = [deal(net, 4, entry=1)]
        pos = position(direction=direction, magic=magic)
        rig.pass_at(3910.0 - (1.0 if direction == -1 else -1.0), [pos])
        rig.pass_at(3909.0, [])
        return rig

    def test_a_losing_bot_trade_starts_the_pause_for_its_direction(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        rig = self.end_trade(run_module, monkeypatch, -5.0, direction=-1)
        rig.bot._bar_counter += 1                         # the next closed bar
        assert rig.bot._entry_cooldown_left(-1) == 3
        assert rig.bot._entry_cooldown_left(1) == 0

    def test_a_winning_trade_does_not(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        rig = self.end_trade(run_module, monkeypatch, +5.0)
        assert rig.bot._entry_cooldown_left(-1) == 0

    def test_a_manual_trade_never_triggers_it(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        rig = self.end_trade(run_module, monkeypatch, -50.0, magic=0)
        assert rig.bot._entry_cooldown_left(-1) == 0

    def test_it_is_off_by_default(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["reentry_cooldown_bars"] == 0
        rig = Rig(run_module, monkeypatch)                # no reentry_cooldown_bars override
        rig.deals = [deal(-5.0, 4, entry=1)]
        rig.pass_at(3909.0, [position()])
        rig.pass_at(3909.0, [])
        assert rig.bot._entry_cooldown_left(-1) == 0

    def test_without_broker_history_the_last_sampled_move_decides(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        losing = self.end_trade(run_module, monkeypatch, 0.0, deals=False, direction=1)
        assert losing.bot._entry_cooldown_left(1) == 0     # last move was +1.0 for a BUY: a gain
        rig = Rig(run_module, monkeypatch, reentry_cooldown_bars=3)
        rig.pass_at(3912.0, [position(direction=-1)])      # a SELL from 3910 at 3912: -2.0
        rig.pass_at(3912.0, [])
        rig.bot._bar_counter += 1                          # asked at the next closed bar, as the bot does
        assert rig.bot._entry_cooldown_left(-1) == 3

    def test_the_setting_is_declared_in_the_schema(self, run_and_stub):
        spec = PARAMS["reentry_cooldown_bars"]
        assert spec.default == 0 and spec.min == 0 and spec.env == "REENTRY_COOLDOWN_BARS"


class TestWiredIntoTheEntryPath:
    def bot(self, run_module, monkeypatch, **cfg):
        c = dict(run_module.CONFIG)
        c.update(dry_run=False, symbol="XAUUSDm", magic_number=BOT_MAGIC, reentry_cooldown_bars=3)
        c.update(cfg)
        bot = run_module.StandaloneAIBot(c)
        self.n = 0
        self.sent = []
        self.signal = (-1, 60.0, 0.7, "signal")

        self.freeze = False

        def get_data():
            if not self.freeze:
                self.n += 1
            idx = pd.DatetimeIndex([pd.Timestamp("2026-11-03 10:00") + pd.Timedelta(minutes=5 * self.n)])
            df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=idx)
            return df, df, df

        monkeypatch.setattr(bot, "_ensure_connected", lambda: True)
        monkeypatch.setattr(bot, "_get_data", get_data)
        monkeypatch.setattr(bot, "_prepared_m15", lambda m: m)
        monkeypatch.setattr(bot, "_prepared_h1", lambda h: h)
        monkeypatch.setattr(bot, "_manage_positions", lambda *a, **k: None)
        monkeypatch.setattr(bot, "_close_opposite_positions", lambda *a, **k: None)
        monkeypatch.setattr(bot, "_get_signal", lambda *a: self.signal)
        monkeypatch.setattr(bot, "_send_order", lambda *a, **k: self.sent.append(a[0]) or True)
        monkeypatch.setattr(run_module, "_prepare", lambda df, cfg: df)
        return bot

    def test_entries_in_the_paused_direction_wait_exactly_n_bars(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, monkeypatch)
        bot.step()                                        # bar 1: free, a SELL goes out
        assert self.sent == [-1]
        bot._cooldown.record_loss(-1, bot._bar_counter, 3)         # that SELL lost
        with caplog.at_level("INFO"):
            for _ in range(3):
                bot.step()                                # bars 2, 3, 4: paused
        assert self.sent == [-1]
        assert sum("cooldown left after a losing SELL" in r.message for r in caplog.records) == 3
        bot.step()                                        # bar 5: free again
        assert self.sent == [-1, -1]

    def test_the_other_direction_is_not_held_back(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, monkeypatch)
        bot.step()
        bot._cooldown.record_loss(-1, bot._bar_counter, 3)
        self.signal = (1, 60.0, 0.7, "signal")           # a BUY signal
        bot.step()
        assert self.sent == [-1, 1]

    def test_off_means_no_pause_even_after_a_loss(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, monkeypatch, reentry_cooldown_bars=0)
        bot.step()
        bot._cooldown.record_loss(-1, bot._bar_counter, 3)
        bot.step()
        assert self.sent == [-1, -1]

    def test_the_bar_counter_only_advances_on_a_new_closed_bar(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, monkeypatch)
        bot.step()
        assert bot._bar_counter == 1 and self.sent == [-1]
        self.freeze = True                                 # the same closed bar again
        bot.step()
        bot.step()
        assert bot._bar_counter == 1 and self.sent == [-1]
