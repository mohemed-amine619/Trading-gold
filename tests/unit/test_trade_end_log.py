"""
tests/unit/test_trade_end_log.py — the TRADE END line: how far each trade ran for
and against us, so break-even and trailing stops can be tuned from evidence.

Price path used (SELL entered at 3910.0, ATR 5.0, profit = entry - price):
  3909 (+1.0)  3906 (+4.0)  3912 (-2.0)  then the position is gone.
  -> peak 4.00 (0.80 ATR), worst -2.00 (0.40 ATR)
"""

from __future__ import annotations

import re
import types

import pandas as pd
import pytest

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401

ENTRY, ATR = 3910.0, 5.0
BOT_MAGIC = 888992


def position(ticket=11, direction=-1, entry=ENTRY, magic=BOT_MAGIC, sl=3919.0):
    return types.SimpleNamespace(
        ticket=ticket, symbol="XAUUSDm", type=1 if direction == -1 else 0, sl=sl, tp=3890.0,
        price_open=entry, volume=0.01, magic=magic, profit=0.0, swap=0.0, time=0)


class Rig:
    def __init__(self, run_module, monkeypatch, **cfg_over):
        self.box = {"positions": []}
        self.deals = None
        mt5 = run_module.mt5
        monkeypatch.setattr(mt5, "positions_get", lambda **k: list(self.box["positions"]))
        monkeypatch.setattr(mt5, "history_deals_get", lambda **k: self.deals, raising=False)
        monkeypatch.setattr(mt5, "order_send", lambda req: types.SimpleNamespace(retcode=10009))
        cfg = dict(run_module.CONFIG)
        cfg.update(dry_run=False, magic_number=BOT_MAGIC, dynamic_sltp_enabled=False,
                   manual_intel_enabled=False, symbol="XAUUSDm", structure_trailing=False,
                   enable_breakeven=False, enable_partial_close=False)
        cfg.update(cfg_over)
        self.bot = run_module.StandaloneAIBot(cfg)
        monkeypatch.setattr(self.bot, "_close_position", lambda *a, **k: None)

    def pass_at(self, price, positions=None, atr=ATR):
        if positions is not None:
            self.box["positions"] = positions
        df = pd.DataFrame([{"close": price, "atr": atr, "high": price + 0.3, "low": price - 0.3}])
        tick = types.SimpleNamespace(bid=price - 0.0, ask=price + 0.0)
        self.bot._manage_positions(df, tick=tick, equity=500.0)


def ends(caplog):
    return [r.message for r in caplog.records if "TRADE END" in r.message]


def deal(profit, reason, entry=1, commission=0.0, swap=0.0, time=100):
    return types.SimpleNamespace(profit=profit, commission=commission, swap=swap, fee=0.0,
                                 entry=entry, reason=reason, time=time)


class TestTradeEnd:
    def run_path(self, run_module, monkeypatch, caplog, **kw):
        rig = Rig(run_module, monkeypatch, **kw)
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [position()])
            rig.pass_at(3906.0)
            rig.pass_at(3912.0)
            rig.pass_at(3912.0, [])                      # the position is gone
        return rig

    def test_peak_and_worst_are_recorded_in_price_and_atr(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        self.run_path(run_module, monkeypatch, caplog)
        (line,) = ends(caplog)
        assert "ticket=11" in line and f"magic={BOT_MAGIC}" in line and "SELL XAUUSDm" in line
        assert "entry=3910.00000" in line
        assert "peak=4.00 (0.80 ATR)" in line
        assert "worst=-2.00 (0.40 ATR)" in line
        assert "atr=5.00" in line

    def test_it_is_logged_exactly_once(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = self.run_path(run_module, monkeypatch, caplog)
        with caplog.at_level("INFO"):
            for _ in range(5):
                rig.pass_at(3912.0, [])
        assert len(ends(caplog)) == 1

    def test_the_last_position_of_a_run_is_reported_even_with_no_positions_left(
            self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [position()])
            rig.pass_at(3909.0, [])                      # _manage_positions used to return early here
        assert len(ends(caplog)) == 1

    def test_the_result_and_exit_reason_come_from_the_broker_history(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        rig.deals = [deal(0.0, 3, entry=0, commission=-0.07),            # opening deal
                     deal(-8.59, 4, entry=1, time=200)]                  # closed by the stop
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [position()])
            rig.pass_at(3909.0, [])
        (line,) = ends(caplog)
        assert line.endswith("pnl=-8.66 exit=sl")

    @pytest.mark.parametrize("reason,name", [(1, "mobile"), (0, "client"), (5, "tp"), (3, "expert")])
    def test_exit_reasons_are_named(self, run_and_stub, monkeypatch, caplog, reason, name):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        rig.deals = [deal(1.0, reason, entry=1)]
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [position()])
            rig.pass_at(3909.0, [])
        assert ends(caplog)[0].endswith(f"exit={name}")

    def test_without_history_the_line_still_appears_without_the_result(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        self.run_path(run_module, monkeypatch, caplog)
        assert "pnl=" not in ends(caplog)[0]

    def test_a_failing_history_call_does_not_break_anything(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        monkeypatch.setattr(run_module.mt5, "history_deals_get",
                            lambda **k: (_ for _ in ()).throw(RuntimeError("terminal busy")), raising=False)
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [position()])
            rig.pass_at(3909.0, [])
        assert len(ends(caplog)) == 1 and "pnl=" not in ends(caplog)[0]

    def test_manual_trades_are_reported_with_their_own_magic_number(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch, ai_protect_manual_trades=False,
                  smart_manage_manual_trades=False)
        manual = position(ticket=22, magic=0)
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [manual])
            rig.pass_at(3909.0, [])
        assert "magic=0" in ends(caplog)[0]

    def test_two_trades_are_tracked_separately(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch, ai_protect_manual_trades=False,
                  smart_manage_manual_trades=False, max_pos_per_symbol=2)
        a, b = position(ticket=1), position(ticket=2, direction=1, entry=3900.0, sl=3890.0)
        with caplog.at_level("INFO"):
            rig.pass_at(3905.0, [a, b])
            rig.pass_at(3902.0)
            rig.pass_at(3902.0, [b])                    # only trade 1 ended
        (line,) = ends(caplog)
        assert "ticket=1 " in line and "peak=8.00" in line        # a SELL from 3910 to 3902

    def test_a_buy_measures_excursion_in_its_own_direction(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        buy = position(direction=1, entry=3900.0, sl=3890.0)
        with caplog.at_level("INFO"):
            rig.pass_at(3903.0, [buy])                   # +3
            rig.pass_at(3897.0)                          # -3
            rig.pass_at(3897.0, [])
        line = ends(caplog)[0]
        assert "BUY" in line and "peak=3.00 (0.60 ATR)" in line and "worst=-3.00 (0.60 ATR)" in line

    def test_nothing_is_logged_for_a_trade_that_is_still_open(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        with caplog.at_level("INFO"):
            for price in (3909.0, 3908.0, 3907.0):
                rig.pass_at(price, [position()])
        assert ends(caplog) == []

    def test_modified_stops_are_counted(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        rig = Rig(run_module, monkeypatch)
        pos = position()
        with caplog.at_level("INFO"):
            rig.pass_at(3909.0, [pos])
            rig.bot._modify_sl(pos, 3915.0)
            rig.bot._modify_sl(pos, 3914.0)
            rig.pass_at(3909.0, [])
        assert "sl_mods=2" in ends(caplog)[0]

    def test_the_line_matches_the_documented_format(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        self.run_path(run_module, monkeypatch, caplog)
        pattern = (r"TRADE END ticket=\d+ magic=\d+ (BUY|SELL) \S+ entry=[\d.]+ peak=[\d.]+ "
                   r"\([\d.]+ ATR\) worst=-?[\d.]+ \([\d.]+ ATR\) atr=[\d.]+ sl_mods=\d+ seen=\d+s")
        assert re.search(pattern, ends(caplog)[0])
