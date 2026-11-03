"""
tests/unit/test_safety_guards.py — two guards that protect against mistakes made
before the bot has even looked at the market:

* the first closed bar after a (re)start is not traded: a restart on a closed
  market used to act on a bar two days old;
* REQUIRE_DEMO_ACCOUNT: refuse to run on anything but a demo account, at connect,
  on reconnect and on every step, and fail closed when the type cannot be read.

MT5's ACCOUNT_TRADE_MODE: 0 demo, 1 contest, 2 real.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from core.config.schema import PARAMS
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401

DEMO, CONTEST, REAL = 0, 1, 2


class Stepper:
    """Drives step() with everything around the entry decision stubbed out."""

    def __init__(self, run_module, monkeypatch, **cfg):
        c = dict(run_module.CONFIG)
        c.update(dry_run=False, symbol="XAUUSDm")
        c.update(cfg)
        self.bot = run_module.StandaloneAIBot(c)
        self.n = 0
        self.sent, self.closed = [], []
        self.signal = (-1, 60.0, 0.7, "signal")

        def get_data():
            self.n += 1
            idx = pd.DatetimeIndex([pd.Timestamp("2026-11-01 22:00") + pd.Timedelta(minutes=5 * self.n)])
            df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=idx)
            return df, df, df

        b = self.bot
        monkeypatch.setattr(b, "_ensure_connected", lambda: True)
        monkeypatch.setattr(b, "_get_data", get_data)
        monkeypatch.setattr(b, "_prepared_m15", lambda m: m)
        monkeypatch.setattr(b, "_prepared_h1", lambda h: h)
        monkeypatch.setattr(b, "_manage_positions", lambda *a, **k: None)
        monkeypatch.setattr(b, "_close_opposite_positions", lambda *a, **k: self.closed.append(a))
        monkeypatch.setattr(b, "_get_signal", lambda *a: self.signal)
        monkeypatch.setattr(b, "_send_order", lambda *a, **k: self.sent.append(a[0]) or True)
        monkeypatch.setattr(run_module, "_prepare", lambda df, cfg: df)


class TestFirstBarAfterStart:
    def test_the_first_closed_bar_is_not_traded_but_the_next_one_is(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch)
        with caplog.at_level("INFO"):
            s.bot.step()                                  # first bar: a SELL signal, possibly stale
        assert s.sent == [] and s.closed == []
        assert any("First bar after start is not traded" in r.message for r in caplog.records)
        s.bot.step()                                      # a bar that closed while running
        assert s.sent == [-1]

    def test_it_also_does_not_close_positions_on_the_stale_signal(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch)
        s.signal = (1, 70.0, 0.8, "signal")               # would close the opposite positions
        s.bot.step()
        assert s.closed == []
        s.bot.step()
        assert len(s.closed) == 1

    def test_it_can_be_switched_off(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch, skip_first_bar=False)
        s.bot.step()
        assert s.sent == [-1]

    def test_it_applies_only_once_per_start_not_to_every_bar(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch)
        for _ in range(4):
            s.bot.step()
        assert len(s.sent) == 3                           # bars 2, 3 and 4

    def test_the_bar_is_still_evaluated_and_counted(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch)
        s.bot.step()
        assert s.bot._bar_counter == 1 and s.bot.last_bar_time is not None

    def test_a_flat_first_bar_is_also_just_skipped_quietly(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch)
        s.signal = (0, 20.0, 0.5, "score 20 < threshold")
        s.bot.step()
        s.bot.step()
        assert s.sent == []

    def test_default_is_on_and_declared(self, run_and_stub):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["skip_first_bar"] is True
        assert PARAMS["skip_first_bar"].default is True and PARAMS["skip_first_bar"].env == "SKIP_FIRST_BAR"


def account(login=261009880, trade_mode=DEMO, **extra):
    acc = types.SimpleNamespace(login=login, balance=500.0, equity=500.0, margin=0.0,
                                margin_free=500.0, **extra)
    if trade_mode is not None:
        acc.trade_mode = trade_mode
    return acc


class TestDemoOnlyGuard:
    def connect(self, run_module, monkeypatch, tmp_path, acc, require=True, login=261009880):
        mt5 = run_module.mt5
        monkeypatch.setattr(mt5, "initialize", lambda **k: True, raising=False)
        monkeypatch.setattr(mt5, "account_info", lambda: acc)
        monkeypatch.setattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", DEMO, raising=False)
        cfg = dict(run_module.CONFIG)
        cfg.update(symbol="XAUUSDm", account=login, require_demo_account=require)
        bot = run_module.StandaloneAIBot(cfg)
        bot._risk_state_path = str(tmp_path / "risk.json")
        return bot, bot.connect()

    def test_a_demo_account_is_accepted(self, run_and_stub, monkeypatch, tmp_path):
        run_module, _ = run_and_stub
        assert self.connect(run_module, monkeypatch, tmp_path, account(trade_mode=DEMO))[1] is True

    @pytest.mark.parametrize("mode", [REAL, CONTEST])
    def test_a_non_demo_account_is_refused(self, run_and_stub, monkeypatch, tmp_path, caplog, mode):
        run_module, _ = run_and_stub
        with caplog.at_level("ERROR"):
            bot, ok = self.connect(run_module, monkeypatch, tmp_path, account(trade_mode=mode))
        assert ok is False
        assert any("is not a demo account" in r.message and "Refusing to trade" in r.message
                   for r in caplog.records)

    def test_an_account_whose_type_cannot_be_read_is_refused(self, run_and_stub, monkeypatch, tmp_path):
        run_module, _ = run_and_stub
        assert self.connect(run_module, monkeypatch, tmp_path, account(trade_mode=None))[1] is False

    def test_a_real_account_is_fine_when_the_guard_is_off(self, run_and_stub, monkeypatch, tmp_path):
        run_module, _ = run_and_stub
        assert self.connect(run_module, monkeypatch, tmp_path, account(trade_mode=REAL), require=False)[1] is True

    def test_it_is_off_by_default_and_marked_unsafe_to_change_live(self, run_and_stub):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["require_demo_account"] is False
        spec = PARAMS["require_demo_account"]
        assert spec.default is False and spec.env == "REQUIRE_DEMO_ACCOUNT"
        assert spec.requires_restart is True and spec.safe_live is False

    def test_a_refused_connect_leaves_the_bot_disconnected(self, run_and_stub, monkeypatch, tmp_path):
        run_module, _ = run_and_stub
        bot, ok = self.connect(run_module, monkeypatch, tmp_path, account(trade_mode=REAL))
        assert ok is False and bot.status.get("connected") is False

    def test_the_guard_rechecks_on_every_step(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch, require_demo_account=True, skip_first_bar=False)
        current = {"acc": account(trade_mode=DEMO)}
        monkeypatch.setattr(run_module.mt5, "account_info", lambda: current["acc"])
        monkeypatch.setattr(run_module.mt5, "ACCOUNT_TRADE_MODE_DEMO", DEMO, raising=False)
        monkeypatch.setattr(run_module.mt5, "symbol_info_tick",
                            lambda sym: types.SimpleNamespace(bid=1.0, ask=1.1))
        s.bot.step()
        assert s.sent == [-1]                                      # on the demo it trades
        current["acc"] = account(trade_mode=REAL)                  # someone switches the terminal
        with caplog.at_level("ERROR"):
            for _ in range(3):
                s.bot.step()
        assert s.sent == [-1]                                      # no new orders
        assert sum("non-demo account" in r.message for r in caplog.records) == 1   # logged once a minute

    def test_a_real_account_trades_normally_when_the_guard_is_off(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        s = Stepper(run_module, monkeypatch, require_demo_account=False, skip_first_bar=False)
        monkeypatch.setattr(run_module.mt5, "account_info", lambda: account(trade_mode=REAL))
        monkeypatch.setattr(run_module.mt5, "symbol_info_tick",
                            lambda sym: types.SimpleNamespace(bid=1.0, ask=1.1))
        s.bot.step()
        assert s.sent == [-1]
