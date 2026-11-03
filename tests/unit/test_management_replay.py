"""
tests/unit/test_management_replay.py — re-running real entries under different
break-even / trailing settings.

All paths are hand-built: long from 100.0, ATR 2.0, stop 96.0, target 108.0 unless
stated. Break-even buffer 0.15 ATR = 0.30, so break-even is a stop at 100.30; the
legacy distance behind price is 0.55 ATR = 1.10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtesting.management_replay import (
    ManagementParams, ReplayTrade, calibration, paired_difference, run_replay, simulate_trade,
    summarise, usd, with_params,
)

T0 = pd.Timestamp("2026-11-03 10:00")
# isolate break-even: no early protection, no trailing, no giveback
BE_ONLY = ManagementParams(early_protect_atr=99.0, trail_activate_atr=99.0, profit_lock_activate_atr=99.0)


def bars(rows, start=T0):
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def long_trade(entry=100.0, sl=96.0, tp=108.0, atr=2.0, volume=0.01, ticket="1", actual=None):
    return ReplayTrade(ticket, 1, T0, entry, sl, tp, volume, atr, actual)


def short_trade(entry=100.0, sl=104.0, tp=92.0, atr=2.0, volume=0.01, ticket="1"):
    return ReplayTrade(ticket, -1, T0, entry, sl, tp, volume, atr)


class TestExits:
    def test_the_stop_is_hit(self):
        r = simulate_trade(long_trade(), bars([(100, 100.5, 99.5, 100.2), (100, 100.2, 95.0, 95.5)]), BE_ONLY)
        assert (r.reason, r.exit_price, r.move) == ("sl", 96.0, -4.0)

    def test_the_target_is_hit(self):
        r = simulate_trade(long_trade(), bars([(100, 100.5, 99.5, 100.2), (100.2, 108.5, 100.0, 108.2)]), BE_ONLY)
        assert (r.reason, r.exit_price, r.move) == ("tp", 108.0, 8.0)

    def test_a_bar_touching_both_takes_the_stop_pessimistically(self):
        r = simulate_trade(long_trade(), bars([(100, 109.0, 95.0, 100.0)]), BE_ONLY)
        assert r.reason == "sl" and r.move == -4.0

    def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop(self):
        r = simulate_trade(long_trade(), bars([(100, 100.2, 99.9, 100.0), (94.0, 94.5, 93.0, 93.5)]), BE_ONLY)
        assert (r.reason, r.exit_price, r.move) == ("sl", 94.0, -6.0)

    def test_a_short_mirrors_a_long(self):
        r = simulate_trade(short_trade(), bars([(100, 105.0, 99.9, 104.5)]), BE_ONLY)
        assert (r.reason, r.exit_price, r.move) == ("sl", 104.0, -4.0)
        r = simulate_trade(short_trade(), bars([(100, 100.1, 91.5, 92.5)]), BE_ONLY)
        assert (r.reason, r.exit_price, r.move) == ("tp", 92.0, 8.0)

    def test_the_spread_is_charged_once(self):
        r = simulate_trade(long_trade(), bars([(100, 100.2, 95.0, 95.5)]), BE_ONLY, spread=0.3)
        assert r.move == pytest.approx(-4.3)

    def test_a_trade_that_never_ends_closes_at_the_last_close(self):
        r = simulate_trade(long_trade(), bars([(100, 100.4, 99.8, 100.1)] * 5), BE_ONLY, max_bars=3)
        assert r.reason == "timeout" and r.exit_price == 100.1

    def test_bars_before_the_entry_are_ignored(self):
        before = bars([(100, 120.0, 80.0, 100.0)] * 3, start=T0 - pd.Timedelta(minutes=5))
        after = bars([(100, 100.4, 99.8, 100.1)] * 3, start=T0)
        r = simulate_trade(long_trade(), pd.concat([before, after]), BE_ONLY, max_bars=3)
        assert r.reason == "timeout"                        # the huge earlier bars did not stop it out

    def test_the_peak_is_recorded(self):
        r = simulate_trade(long_trade(), bars([(100, 103.0, 99.9, 102.0), (102, 102.2, 95.0, 95.5)]), BE_ONLY)
        assert r.peak == pytest.approx(3.0)


class TestBreakEven:
    def test_legacy_distance_moves_the_stop_to_break_even_at_about_0_7_atr(self):
        # bar 1 closes at 101.5 (profit 1.5): target min(100.65, 101.5 - 1.10 = 100.40) = 100.40 >= 100.30
        # so the stop becomes 100.40 for the next bar, which dips to 100.1 and is stopped there
        path = bars([(100, 101.5, 99.9, 101.5), (101.5, 101.8, 100.1, 100.2)])
        r = simulate_trade(long_trade(), path, BE_ONLY)
        assert (r.reason, r.exit_price) == ("sl", pytest.approx(100.4)) and r.move == pytest.approx(0.4)
        assert r.stop_moves == 1

    def test_a_wider_distance_lets_the_same_trade_run_to_its_target(self):
        # lock distance 1.0 ATR = 2.0: target min(100.65, 99.5) = 99.5 < break-even 100.30, so no
        # break-even yet; the dip to 100.1 is survived and bar 3 reaches the target
        path = bars([(100, 101.5, 99.9, 101.5), (101.5, 101.8, 100.1, 100.2), (100.2, 108.5, 100.0, 108.0)])
        wide = with_params(BE_ONLY, lock_min_distance_atr=1.0)
        r = simulate_trade(long_trade(), path, wide)
        assert (r.reason, r.move) == ("tp", 8.0) and r.stop_moves == 0
        legacy = simulate_trade(long_trade(), path, BE_ONLY)
        assert legacy.move == pytest.approx(0.4)            # the same entry, +0.40 vs +8.00

    def test_an_explicit_055_equals_the_legacy_default(self):
        path = bars([(100, 101.5, 99.9, 101.5), (101.5, 101.8, 100.1, 100.2)])
        a = simulate_trade(long_trade(), path, BE_ONLY)
        b = simulate_trade(long_trade(), path, with_params(BE_ONLY, lock_min_distance_atr=0.55))
        assert a == b

    def test_a_non_gold_symbol_uses_the_0_40_legacy_distance(self):
        assert with_params(BE_ONLY, is_gold=False).distance_atr() == 0.40
        assert BE_ONLY.distance_atr() == 0.55
        assert with_params(BE_ONLY, lock_min_distance_atr=0.9).distance_atr() == 0.9

    def test_break_even_can_be_switched_off(self):
        path = bars([(100, 101.5, 99.9, 101.5), (101.5, 101.8, 100.1, 100.2), (100.2, 100.4, 99.0, 99.5)])
        r = simulate_trade(long_trade(), path, with_params(BE_ONLY, enable_breakeven=False))
        assert r.stop_moves == 0 and r.reason == "timeout"

    def test_the_stop_never_loosens(self):
        # rises (stop moves up), then falls back: the stop must not follow price down
        rows = [(100, 101.5, 99.9, 101.5), (101.5, 103.0, 101.2, 102.8), (102.8, 103.0, 100.9, 101.0)]
        up = simulate_trade(long_trade(), bars(rows), BE_ONLY)
        assert up.reason == "sl" and up.move > 0           # stopped in profit, never below break-even


class TestGiveback:
    def test_a_trade_that_gives_back_too_much_of_its_peak_is_closed(self):
        # peak 3.0 >= 2.0 activates it; bar 1 closes at 102.0 (profit 2.0 > 1.65, kept); bar 2 closes
        # at 101.6 (profit 1.6 <= 3.0 * 0.55 = 1.65) -> closed there
        params = with_params(BE_ONLY, profit_lock_activate_atr=1.0, profit_giveback_pct=0.45)
        path = bars([(100, 103.0, 99.9, 102.0), (102, 102.2, 101.2, 101.6), (101.6, 101.7, 101.2, 101.5)])
        r = simulate_trade(long_trade(), path, params)
        assert (r.reason, r.exit_price) == ("giveback", 101.6) and r.move == pytest.approx(1.6)


class TestAggregates:
    def test_usd(self):
        assert usd(1.4, 0.01) == pytest.approx(1.4)
        assert usd(-4.44, 0.02, contract_size=100.0) == pytest.approx(-8.88)

    def test_summary_hand_computed(self):
        rows = [{"pnl": 2.0}, {"pnl": 2.0}, {"pnl": 2.0}, {"pnl": -4.0}, {"pnl": -4.0}]
        s = summarise(rows)
        assert s["trades"] == 5 and s["net"] == -2.0 and s["win_rate"] == 0.6
        assert s["avg_win"] == 2.0 and s["avg_loss"] == -4.0
        assert s["profit_factor"] == 0.75 and s["expectancy"] == -0.4

    def test_summary_of_nothing_and_of_all_wins(self):
        assert summarise([])["win_rate"] is None
        s = summarise([{"pnl": 1.0}, {"pnl": 2.0}])
        assert s["profit_factor"] is None and s["avg_loss"] is None

    def test_paired_difference_hand_computed(self):
        a = [{"ticket": str(i), "pnl": 0.0} for i in range(4)]
        b = [{"ticket": str(i), "pnl": float(i + 1)} for i in range(4)]       # diffs 1, 2, 3, 4
        d = paired_difference(a, b)
        assert d["n"] == 4 and d["mean"] == 2.5
        assert d["se"] == pytest.approx(1.29099 / 2, abs=0.001) and d["z"] == pytest.approx(3.87, abs=0.01)

    def test_paired_difference_uses_only_common_entries(self):
        a = [{"ticket": "1", "pnl": 1.0}, {"ticket": "2", "pnl": 2.0}]
        b = [{"ticket": "2", "pnl": 3.0}, {"ticket": "9", "pnl": 9.0}]
        assert paired_difference(a, b)["n"] == 1 and paired_difference(a, b)["mean"] is None

    def test_identical_sets_have_no_difference(self):
        a = [{"ticket": str(i), "pnl": float(i)} for i in range(5)]
        d = paired_difference(a, a)
        assert d["mean"] == 0.0 and d["z"] is None

    def test_calibration_perfect_and_poor(self):
        rows = [{"pnl": float(x), "actual_pnl": float(x)} for x in (1, -2, 3, -4)]
        c = calibration(rows)
        assert c["mean_abs_error"] == 0.0 and c["correlation"] == 1.0
        assert c["replayed_net"] == c["actual_net"] == -2.0
        assert calibration([{"pnl": 1.0, "actual_pnl": None}])["correlation"] is None

    def test_unknown_parameters_are_refused(self):
        with pytest.raises(ValueError, match="unknown management parameter"):
            with_params(BE_ONLY, breakeven_trigger=0.5)

    def test_run_replay_carries_the_actual_result_for_calibration(self):
        t = long_trade(actual=-3.7)
        rows = run_replay([t], bars([(100, 100.2, 95.0, 95.5)]), BE_ONLY)
        assert rows[0]["pnl"] == pytest.approx(-4.0) and rows[0]["actual_pnl"] == -3.7
        assert rows[0]["reason"] == "sl"


def _load_cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location("replay_cli", "scripts/replay_management.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class TestInputsAndCli:
    def _m5(self, n=40, rng=2.0):
        idx = pd.date_range("2026-11-03 08:00", periods=n, freq="5min")
        return pd.DataFrame({"open": 100.0, "high": 100.0 + rng / 2, "low": 100.0 - rng / 2,
                             "close": 100.0}, index=idx)

    def test_atr_is_the_mean_true_range(self):
        from core.backtesting.management_replay import atr_series
        a = atr_series(self._m5()).dropna()
        assert a.iloc[-1] == pytest.approx(2.0) and len(a) == 40 - 13

    def test_entries_are_skipped_not_guessed(self):
        from core.backtesting.management_replay import build_replay_trades

        def e(**k):
            return {"time": "2026-11-03 10:00", "side": "BUY", "price": 100.0, "sl": 96.0,
                    "tp": 108.0, "volume": 0.01, "ticket": "7", **k}
        entries = [e(), e(ticket=None), e(time="2026-11-03 08:05", ticket="9"), e(sl=0.0, ticket="10"),
                   e(side="SELL", ticket="8")]
        trades, skipped = build_replay_trades(entries, self._m5(), {"7": 1.5})
        assert [t.ticket for t in trades] == ["7", "8"]
        assert (trades[0].direction, trades[1].direction) == (1, -1)
        assert trades[0].atr == pytest.approx(2.0)
        assert trades[0].actual_pnl == 1.5 and trades[1].actual_pnl is None
        assert skipped == {"no_ticket": 1, "no_atr": 1, "no_stop": 1}

    def test_the_log_time_shift_is_applied(self):
        from core.backtesting.management_replay import build_replay_trades
        entry = {"time": "2026-11-03 07:00", "side": "BUY", "price": 100.0, "sl": 96.0, "tp": 108.0,
                 "volume": 0.01, "ticket": "7"}
        trades, _ = build_replay_trades([entry], self._m5(), shift_hours=3.0)
        assert trades[0].entry_time == pd.Timestamp("2026-11-03 10:00")

    def test_cli_end_to_end_and_refuses_fine_bars_without_atr_bars(self, tmp_path, capsys):
        cli = _load_cli()
        self._m5().to_csv(tmp_path / "m5.csv")
        idx = pd.date_range("2026-11-03 10:00", periods=30, freq="1min")
        px = [100 + 0.3 * i for i in range(30)]
        pd.DataFrame({"open": px, "high": [p + 0.2 for p in px], "low": [p - 0.2 for p in px],
                      "close": px}, index=idx).to_csv(tmp_path / "m1.csv")
        (tmp_path / "bot.log").write_text(
            "2026-11-03 10:00:05,000 [INFO] ✅ BUY XAUUSDm vol=0.01 @ 100.00  SL=96.00  TP=108.00 "
            "ticket=7 conf=0.60 score=60.0\n", encoding="utf-8")
        (tmp_path / "t.csv").write_text("position_id,net_pnl\n7,1.1\n", encoding="utf-8")
        base = ["--log", str(tmp_path / "bot.log"), "--bars", str(tmp_path / "m1.csv")]

        assert cli.main(base) == 2                                  # M1 bars, no --atr-bars
        assert "atr-bars" in capsys.readouterr().out
        assert cli.main(base + ["--atr-bars", str(tmp_path / "m5.csv"), "--trades", str(tmp_path / "t.csv"),
                                "--set", "mine=lock_min_distance_atr:1.5"]) == 0
        out = capsys.readouterr().out
        assert "1 replayable" in out and "CALIBRATION" in out and "mine" in out and "no-breakeven" in out

    def test_parse_set(self):
        assert _load_cli().parse_set("x=lock_min_distance_atr:1.5,enable_breakeven:false") == (
            "x", {"lock_min_distance_atr": 1.5, "enable_breakeven": False})
