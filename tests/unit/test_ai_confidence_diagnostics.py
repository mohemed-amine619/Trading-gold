"""
tests/unit/test_ai_confidence_diagnostics.py — does the AI's confidence separate
winners from losers? Entry lines now carry the ticket and confidence; the
diagnostics join them to the broker's closed trades.
"""

from __future__ import annotations

import json
import types

import pytest

from core.diagnostics.confidence import (
    BUCKETS, MIN_MATCHED_FOR_FINDING, confidence_buckets, confidence_finding,
)
from core.diagnostics.logstats import parse_log
from core.diagnostics.report import build_summary, findings
from core.diagnostics.trades import Trade, trade_stats
from tests.unit.test_collect_diagnostics_cli import (  # noqa: F401
    FakeMT5, NOW, bot_dir, cli, members, run,
)
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401


def trade(position_id, pnl):
    return Trade(position_id=position_id, symbol="XAUUSDm", direction=-1, volume=0.01,
                 open_time="2026-11-02 08:00:00", close_time="2026-11-02 08:30:00",
                 open_price=1.0, close_price=1.0, net_pnl=pnl, exit_reason="sl",
                 duration_s=1800, magic=1, comment="")


def entry(ticket, conf):
    return {"ticket": ticket, "conf": conf, "score": 60.0, "time": "2026-11-02 08:00"}


class TestBuckets:
    def setup_method(self):
        # position -> net: 101:-5, 102:-5, 103:+2, 104:-4, 105:-6, 106:-7
        self.trades = [trade(101, -5), trade(102, -5), trade(103, 2), trade(104, -4),
                       trade(105, -6), trade(106, -7)]
        self.entries = [entry("101", 0.52), entry("102", 0.53), entry("103", 0.54),
                        entry("104", 0.60), entry("105", 0.70), entry("106", 0.78),
                        {"ticket": None, "conf": None, "score": None},      # old log format
                        entry("999", 0.66)]                                 # no closed trade
        self.result = confidence_buckets(self.entries, self.trades)

    def test_counts_of_matched_unmatched_and_untagged(self):
        r = self.result
        assert (r["matched"], r["unmatched"], r["untagged"]) == (6, 1, 1)

    def test_the_lowest_bucket_hand_computed(self):
        b = self.result["buckets"]["<0.55"]
        assert b["trades"] == 3 and b["wins"] == 1
        assert b["win_rate"] == pytest.approx(1 / 3, abs=0.001)
        assert b["net"] == -8.0
        assert b["avg_net"] == pytest.approx(-8 / 3, abs=0.001)

    def test_other_buckets(self):
        b = self.result["buckets"]
        assert b["0.55-0.65"]["trades"] == 1 and b["0.55-0.65"]["net"] == -4.0
        assert b["0.65-0.75"]["trades"] == 1 and b["0.65-0.75"]["net"] == -6.0
        assert b[">=0.75"]["trades"] == 1 and b[">=0.75"]["net"] == -7.0

    def test_empty_buckets_report_none_not_zero(self):
        r = confidence_buckets([entry("101", 0.52)], [trade(101, -5)])
        assert r["buckets"][">=0.75"]["trades"] == 0 and r["buckets"][">=0.75"]["win_rate"] is None

    @pytest.mark.parametrize("conf,name", [(0.0, "<0.55"), (0.549, "<0.55"), (0.55, "0.55-0.65"),
                                           (0.65, "0.65-0.75"), (0.75, ">=0.75"), (0.999, ">=0.75")])
    def test_bucket_edges(self, conf, name):
        r = confidence_buckets([entry("1", conf)], [trade(1, 1.0)])
        assert r["buckets"][name]["trades"] == 1

    def test_a_ticket_is_matched_as_text_or_number(self):
        r = confidence_buckets([entry(101, 0.52)], [trade(101, -5)])
        assert r["matched"] == 1

    def test_nothing_in_nothing_out(self):
        r = confidence_buckets([], [])
        assert r["matched"] == 0 and all(b["trades"] == 0 for b in r["buckets"].values())


def sample(low_avg, high_avg, n=6):
    """n trades at low confidence and n at high, with the given per-trade net."""
    trades, entries = [], []
    for i in range(1, n + 1):                      # ticket 0 means "unknown", so start at 1
        trades += [trade(i, low_avg), trade(100 + i, high_avg)]
        entries += [entry(str(i), 0.52), entry(str(100 + i), 0.80)]
    return confidence_buckets(entries, trades)


class TestFinding:
    def test_higher_confidence_paying_worse_is_flagged_with_its_numbers(self):
        note = confidence_finding(sample(low_avg=1.0, high_avg=-1.0))
        assert note and "did not pay better" in note and "-1.0" in note and "n=6" in note
        assert "12 trades matched" in note

    def test_higher_confidence_paying_better_is_not_flagged(self):
        assert confidence_finding(sample(low_avg=-1.0, high_avg=1.0)) is None

    def test_too_few_matched_trades_gives_no_finding(self):
        assert MIN_MATCHED_FOR_FINDING > 5
        assert confidence_finding(sample(1.0, -1.0, n=2)) is None

    def test_a_single_populated_bucket_gives_no_finding(self):
        entries = [entry(str(i), 0.52) for i in range(15)]
        trades = [trade(i, -1.0) for i in range(15)]
        assert confidence_finding(confidence_buckets(entries, trades)) is None

    def test_no_data_gives_no_finding(self):
        assert confidence_finding(None) is None

    def test_the_finding_reaches_the_findings_list(self):
        conf = sample(1.0, -1.0)
        assert any("did not pay better" in f for f in findings(trade_stats([]), parse_log([]), conf))


class TestLogParsing:
    TAGGED = ("2026-11-02 08:00:00,802 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3906.75100  SL=3915.34300  "
              "TP=3889.56700  ticket=2159773984 conf=0.686 score=62.8\n")
    OLD = ("2026-11-02 08:05:00,802 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3907.64400  SL=3916.25400  "
           "TP=3890.42400\n")
    PROMOTED = ("2026-11-02 12:25:15,714 [INFO] [AI] XAUUSDm promoted run #6 | train=623 | oos_acc=0.635 "
                "| auc=0.657 | base=0.436 | lift=+0.071 | calibrated=True | top: ema50_200_spread=0.063\n")

    def test_a_tagged_entry_carries_ticket_confidence_and_score(self):
        (e,) = parse_log([self.TAGGED])["entries"]
        assert e["ticket"] == "2159773984" and e["conf"] == 0.686 and e["score"] == 62.8
        assert e["side"] == "SELL" and e["volume"] == 0.01

    def test_an_old_format_entry_still_parses_without_the_tags(self):
        (e,) = parse_log([self.OLD])["entries"]
        assert e["ticket"] is None and e["conf"] is None and e["price"] == 3907.644

    def test_both_formats_can_share_a_log(self):
        assert len(parse_log([self.OLD, self.TAGGED])["entries"]) == 2

    def test_a_promoted_model_line_is_captured(self):
        (m,) = parse_log([self.PROMOTED])["ai_models"]
        assert m["run"] == 6 and m["train"] == 623
        assert (m["oos_acc"], m["auc"], m["base"], m["lift"]) == (0.635, 0.657, 0.436, 0.071)


class TestSummarySection:
    def test_the_table_is_present_with_matched_trades(self):
        conf = sample(1.0, -1.0)
        md = build_summary({"generated": "now", "days": 7}, trade_stats([]), parse_log([]), conf=conf)
        assert "## AI confidence vs results" in md and "0.55-0.65" in md and ">=0.75" in md
        assert "12 trades matched" in md

    def test_the_empty_case_explains_itself(self):
        md = build_summary({"generated": "now", "days": 7}, trade_stats([]), parse_log([]))
        assert "No trades could be matched" in md

    def test_model_quality_lines_appear(self):
        log = parse_log([TestLogParsing.PROMOTED])
        md = build_summary({"generated": "now", "days": 7}, trade_stats([]), log)
        assert "AI model quality" in md and "#6" in md and "0.635" in md


class TestEntryMessageFromRunPy:
    def bot(self, run_module):
        cfg = dict(run_module.CONFIG)
        cfg.update(dry_run=False, symbol="XAUUSDm")
        b = run_module.StandaloneAIBot(cfg)
        b.status["last_score"] = 62.8
        return b

    def test_the_message_carries_ticket_confidence_and_score(self, run_and_stub):
        run_module, _ = run_and_stub
        msg = self.bot(run_module)._entry_message(
            -1, "XAUUSDm", 0.01, 3906.751, 3915.343, 3889.567,
            types.SimpleNamespace(order=2159773984), 0.686)
        assert msg.endswith("ticket=2159773984 conf=0.686 score=62.8")
        assert msg.startswith("✅ SELL XAUUSDm vol=0.01 @ 3906.75100")

    def test_the_message_round_trips_through_the_log_parser(self, run_and_stub):
        run_module, _ = run_and_stub
        msg = self.bot(run_module)._entry_message(
            1, "XAUUSDm", 0.01, 3959.071, 3951.073, 3975.067, types.SimpleNamespace(order=77), 0.612)
        (e,) = parse_log([f"2026-11-02 18:40:01,384 [INFO] {msg}\n"])["entries"]
        assert (e["side"], e["ticket"], e["conf"], e["score"]) == ("BUY", "77", 0.612, 62.8)

    def test_a_result_without_an_order_id_logs_ticket_zero_not_a_crash(self, run_and_stub):
        run_module, _ = run_and_stub
        msg = self.bot(run_module)._entry_message(
            1, "XAUUSDm", 0.01, 1.0, 0.9, 1.2, types.SimpleNamespace(), 0.6)
        assert "ticket=0" in msg
        assert confidence_buckets(parse_log([f"2026-11-02 18:40:01,384 [INFO] {msg}\n"])["entries"],
                                  [trade(0, 1.0)])["untagged"] == 1

    def test_a_missing_confidence_is_logged_as_zero(self, run_and_stub):
        run_module, _ = run_and_stub
        msg = self.bot(run_module)._entry_message(
            1, "XAUUSDm", 0.01, 1.0, 0.9, 1.2, types.SimpleNamespace(order=5), None)
        assert "conf=0.000" in msg

    def test_the_gate_message_no_longer_rounds_0517_up_to_the_threshold(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[2] / "run.py").read_text(encoding="utf-8")
        assert "gold AI conf {ai_conf:.3f} < threshold ({gold_min_conf:.3f})" in src
        assert "{ai_conf:.2f} < threshold ({gold_min_conf:.2f})" not in src


class TestCollectorEndToEnd:
    def test_entries_are_joined_to_the_brokers_closed_trades(self, cli, bot_dir):
        log = bot_dir / "trading_bot.log"
        log.write_text(log.read_text(encoding="utf-8") + (
            "2026-11-02 08:00:03,000 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3906.75100  SL=3915.34300  "
            "TP=3889.56700  ticket=1 conf=0.700 score=60.0\n"
            "2026-11-02 08:30:03,000 [INFO] ✅ BUY XAUUSDm vol=0.01 @ 3900.00000  SL=3890.00000  "
            "TP=3920.00000  ticket=2 conf=0.560 score=51.0\n"), encoding="utf-8")
        data = json.loads(members(run(cli, bot_dir))["stats.json"])
        conf = data["confidence"]
        assert conf["matched"] == 2 and conf["untagged"] == 1        # the older SELL line has no tags
        assert conf["buckets"]["0.65-0.75"]["trades"] == 1
        assert conf["buckets"]["0.65-0.75"]["net"] == pytest.approx(-8.66)   # position 1
        assert conf["buckets"]["0.55-0.65"]["net"] == pytest.approx(12.0)    # position 2

    def test_the_summary_shows_the_section(self, cli, bot_dir):
        assert "## AI confidence vs results" in members(run(cli, bot_dir))["summary.md"]
