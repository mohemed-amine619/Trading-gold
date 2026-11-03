"""
tests/unit/test_diagnostics_excursion.py — how far trades ran before they ended.

Hand-worked data: eight trades (long, entry 100):
  winners 1-4   net +2   peak (price, ATR): (4.0, 0.8) (6.0, 1.2) (2.5, 0.5) (2.0, 0.4)
                         exit moves: 2.0, 3.0, 1.0, 2.0  -> kept 0.5, 0.5, 0.4, 1.0
  losers  5-8   net -5   peak ATR: 0.6, 0.0, 0.35, 1.1
"""

from __future__ import annotations

import json

import pytest

from core.diagnostics.excursion import (
    MIN_LOSERS_FOR_FINDING, excursion_findings, excursion_report,
)
from core.diagnostics.logstats import parse_log
from core.diagnostics.report import build_summary, findings
from core.diagnostics.trades import Trade, trade_stats
from tests.unit.test_collect_diagnostics_cli import (  # noqa: F401
    FakeMT5, NOW, bot_dir, cli, members, run,
)


def trade(position_id, net, move=0.0):
    return Trade(position_id=position_id, symbol="XAUUSDm", direction=1, volume=0.01,
                 open_time="2026-11-03 10:00:00", close_time="2026-11-03 10:10:00",
                 open_price=100.0, close_price=100.0 + move, net_pnl=float(net),
                 exit_reason="sl", duration_s=600, magic=1, comment="")


def end(ticket, peak, peak_atr, worst_atr=0.2):
    return {"ticket": str(ticket), "peak": peak, "peak_atr": peak_atr, "worst": -worst_atr,
            "worst_atr": worst_atr}


TRADES = [trade(1, 2, 2.0), trade(2, 2, 3.0), trade(3, 2, 1.0), trade(4, 2, 2.0),
          trade(5, -5), trade(6, -5), trade(7, -5), trade(8, -5)]
ENDS = [end(1, 4.0, 0.8), end(2, 6.0, 1.2), end(3, 2.5, 0.5), end(4, 2.0, 0.4),
        end(5, 3.0, 0.6), end(6, 0.0, 0.0), end(7, 1.75, 0.35), end(8, 5.5, 1.1),
        end(999, 1.0, 0.2)]                               # a trade outside the period


class TestReport:
    def setup_method(self):
        self.r = excursion_report(ENDS, TRADES)

    def test_matching_by_ticket(self):
        assert self.r["matched"] == 8 and self.r["unmatched"] == 1

    def test_peak_buckets_hand_computed(self):
        b = self.r["peak_buckets"]
        assert b["<0.3 ATR"]["trades"] == 1 and b["<0.3 ATR"]["wins"] == 0          # trade 6
        assert b["0.3-0.6"]["trades"] == 3 and b["0.3-0.6"]["wins"] == 2            # 3, 4 win; 7 loses
        assert b["0.3-0.6"]["net"] == pytest.approx(2 + 2 - 5)
        assert b["0.6-1.0"]["trades"] == 2                                           # 1 wins, 5 loses
        assert b["0.6-1.0"]["net"] == pytest.approx(2 - 5)
        assert b[">=1.0"]["trades"] == 2 and b[">=1.0"]["net"] == pytest.approx(2 - 5)

    def test_how_many_losers_had_been_in_profit_first(self):
        assert self.r["losers"] == 4
        assert self.r["losers_once_in_profit"] == {"0.3": 3, "0.5": 2, "1": 1}

    def test_winner_peak_and_share_of_the_peak_kept(self):
        assert self.r["winners"] == 4
        assert self.r["winner_avg_peak_atr"] == pytest.approx((0.8 + 1.2 + 0.5 + 0.4) / 4, abs=0.01)
        assert self.r["winner_median_capture"] == pytest.approx(0.5)   # median of 0.5, 0.5, 0.4, 1.0

    def test_losers_average_worst(self):
        assert self.r["loser_avg_worst_atr"] == pytest.approx(0.2)

    def test_a_short_trade_keeps_its_own_sign(self):
        short = Trade(position_id=1, symbol="X", direction=-1, volume=0.01, open_time="2026-11-03 10:00:00",
                      close_time="2026-11-03 10:05:00", open_price=100.0, close_price=97.0,
                      net_pnl=3.0, exit_reason="tp", duration_s=300, magic=1, comment="")
        r = excursion_report([end(1, 6.0, 1.0)], [short])
        assert r["winner_median_capture"] == pytest.approx(0.5)       # moved 3.0 of a 6.0 peak

    def test_no_peak_means_no_capture_ratio_not_a_crash(self):
        r = excursion_report([end(1, 0.0, 0.0)], [trade(1, 1.0, 1.0)])
        assert r["winner_median_capture"] is None

    def test_empty_inputs(self):
        r = excursion_report([], [])
        assert r["matched"] == 0 and r["winner_median_capture"] is None and r["losers"] == 0


def many(n_losers, once_in_profit, n_winners=0, capture=1.0):
    trades, ends = [], []
    for i in range(n_losers):
        trades.append(trade(100 + i, -5))
        ends.append(end(100 + i, 3.0 if i < once_in_profit else 0.5, 0.6 if i < once_in_profit else 0.1))
    for i in range(n_winners):
        trades.append(trade(200 + i, 2, 4.0 * capture))
        ends.append(end(200 + i, 4.0, 0.8))
    return excursion_report(ends, trades)


class TestFindings:
    def test_losers_that_were_in_profit_first_are_flagged_with_numbers(self):
        found = excursion_findings(many(10, once_in_profit=4))
        assert any("4 of 10 losing trades (40%) had first been +0.5 ATR in profit" in f for f in found)
        assert any("it is not proof" in f for f in found)

    def test_a_small_share_is_not_flagged(self):
        assert excursion_findings(many(10, once_in_profit=2)) == []

    def test_too_few_losers_gives_no_finding(self):
        assert MIN_LOSERS_FOR_FINDING > 5
        assert excursion_findings(many(5, once_in_profit=5)) == []

    def test_winners_giving_back_most_of_their_peak_are_flagged(self):
        found = excursion_findings(many(0, 0, n_winners=10, capture=0.3))
        assert any("kept a median 30% of their peak profit" in f for f in found)

    def test_winners_keeping_most_are_not_flagged(self):
        assert excursion_findings(many(0, 0, n_winners=10, capture=0.8)) == []

    def test_nothing_matched_means_no_findings(self):
        assert excursion_findings(None) == [] and excursion_findings(excursion_report([], [])) == []

    def test_findings_reach_the_findings_list(self):
        exc = many(10, once_in_profit=5)
        assert any("losing trades" in f for f in findings(trade_stats([]), parse_log([]), None, exc))


LINE = ("2026-11-03 10:20:00,100 [INFO] 🏁 TRADE END ticket=2167606650 magic=888992 SELL XAUUSDm "
        "entry=3906.75000 peak=4.00 (0.80 ATR) worst=-2.00 (0.40 ATR) atr=5.00 sl_mods=12 seen=212s"
        "{tail}\n")


class TestParsing:
    def test_a_line_without_a_result(self):
        (e,) = parse_log([LINE.format(tail="")])["trade_ends"]
        assert e["ticket"] == "2167606650" and e["magic"] == 888992 and e["side"] == "SELL"
        assert (e["peak"], e["peak_atr"], e["worst"], e["worst_atr"], e["atr"]) == (4.0, 0.8, -2.0, 0.4, 5.0)
        assert e["sl_mods"] == 12 and e["seen_s"] == 212 and e["pnl"] is None and e["exit"] is None

    def test_a_line_with_the_result_and_exit_reason(self):
        (e,) = parse_log([LINE.format(tail=" pnl=-8.66 exit=sl")])["trade_ends"]
        assert e["pnl"] == -8.66 and e["exit"] == "sl"

    def test_a_line_without_an_atr_is_skipped(self):
        bad = LINE.format(tail="").replace("(0.80 ATR)", "(nan ATR)").replace("(0.40 ATR)", "(nan ATR)")
        assert parse_log([bad])["trade_ends"] == []

    def test_other_lines_are_unaffected(self):
        log = parse_log([LINE.format(tail=""),
                         "2026-11-03 10:21:00,000 [INFO] Modified SL ticket=1 → 3900.0\n"])
        assert len(log["trade_ends"]) == 1 and log["sl_modifications"]["total"] == 1


class TestSummary:
    def test_the_section_and_numbers(self):
        md = build_summary({"generated": "now", "days": 7}, trade_stats([]), parse_log([]),
                           excursion=excursion_report(ENDS, TRADES))
        assert "## How far trades ran before they ended" in md
        assert "peak profit reached" in md and "had first reached +0.3 ATR: 3, +0.5 ATR: 2" in md

    def test_the_empty_case_explains_itself(self):
        md = build_summary({"generated": "now", "days": 7}, trade_stats([]), parse_log([]))
        assert "No TRADE END lines matched" in md


class TestCollector:
    def test_trade_end_lines_are_joined_to_the_brokers_closed_trades(self, cli, bot_dir):
        log = bot_dir / "trading_bot.log"
        log.write_text(log.read_text(encoding="utf-8") + (
            "2026-11-02 08:20:00,000 [INFO] 🏁 TRADE END ticket=1 magic=888992 SELL XAUUSDm entry=3906.75 "
            "peak=0.50 (0.10 ATR) worst=-8.00 (1.60 ATR) atr=5.00 sl_mods=3 seen=240s\n"
            "2026-11-02 08:40:00,000 [INFO] 🏁 TRADE END ticket=2 magic=888992 BUY XAUUSDm entry=3900.00 "
            "peak=13.00 (2.60 ATR) worst=-1.00 (0.20 ATR) atr=5.00 sl_mods=9 seen=600s\n"), encoding="utf-8")
        data = json.loads(members(run(cli, bot_dir))["stats.json"])
        exc = data["excursion"]
        assert exc["matched"] == 2 and exc["losers"] == 1 and exc["winners"] == 1
        assert exc["peak_buckets"][">=1.0"]["trades"] == 1                  # position 2 ran 2.6 ATR

    def test_the_summary_shows_the_section(self, cli, bot_dir):
        assert "## How far trades ran before they ended" in members(run(cli, bot_dir))["summary.md"]
