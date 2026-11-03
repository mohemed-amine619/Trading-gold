"""
tests/unit/test_diagnostics_core.py — secret scrubbing, deal pairing, trade
statistics, log statistics and the rule-based findings.

Expected values are worked out by hand. The log samples are real lines from the
bot's own output.
"""

from __future__ import annotations

import pytest

from core.diagnostics.logstats import parse_log
from core.diagnostics.redact import (
    REDACTED, login_from_env, mask_account, redact_env_text, scrub_text, secret_values_from_env,
)
from core.diagnostics.report import build_summary, findings
from core.diagnostics.trades import Trade, pair_deals, trade_stats

ENV = """# Broker
MT5_LOGIN=261009880
MT5_PASSWORD=Hunter2!!
MT5_SERVER=Exness-MT5Trial16
DASHBOARD_TOKEN=abc123def456
MY_API_KEY=xyz9999
RISK_PER_TRADE=0.01
EMPTY_PASSWORD=
"""


class TestRedact:
    def test_secret_values_and_the_login_never_survive(self):
        out = redact_env_text(ENV)
        for leaked in ("Hunter2!!", "abc123def456", "xyz9999", "261009880"):
            assert leaked not in out

    def test_other_lines_are_kept_exactly(self):
        out = redact_env_text(ENV)
        assert "# Broker" in out
        assert "MT5_SERVER=Exness-MT5Trial16" in out
        assert "RISK_PER_TRADE=0.01" in out
        assert "MT5_LOGIN=***9880" in out
        assert f"MT5_PASSWORD={REDACTED}" in out

    def test_an_empty_secret_is_left_alone(self):
        assert "EMPTY_PASSWORD=\n" in redact_env_text(ENV)

    def test_secret_values_are_collected_by_key_name(self):
        assert set(secret_values_from_env(ENV)) == {"Hunter2!!", "abc123def456", "xyz9999"}

    def test_very_short_values_are_not_treated_as_secrets(self):
        assert secret_values_from_env("MT5_PASSWORD=ab\n") == []

    def test_login_is_read_and_masked(self):
        assert login_from_env(ENV) == "261009880"
        assert mask_account("261009880") == "***9880"
        assert mask_account("123") == "***"

    def test_scrub_removes_known_values_wherever_they_appear(self):
        text = "connect failed for Hunter2!! on account 261009880"
        out = scrub_text(text, secrets=["Hunter2!!"], mask_values=["261009880"])
        assert "Hunter2!!" not in out and "261009880" not in out and "***9880" in out

    @pytest.mark.parametrize("line", [
        "login password=SomethingElse99 ok",
        '{"token": "tok_12345678"}',
        "Authorization: Bearer eyJhbGciOi.payload.sig",
        "X-Dashboard-Token: dashval99",
        "api_key: sk-live-0000",
    ])
    def test_scrub_removes_secret_patterns_even_when_the_value_is_unknown(self, line):
        out = scrub_text(line)
        assert REDACTED in out
        for leaked in ("SomethingElse99", "tok_12345678", "eyJhbGciOi", "dashval99", "sk-live-0000"):
            assert leaked not in out

    def test_ordinary_log_text_is_unchanged(self):
        line = "2026-11-02 08:00:00,802 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3906.75100"
        assert scrub_text(line, secrets=["ab"], mask_values=["12"]) == line

    def test_scrubbing_does_not_blank_words_for_tiny_secrets(self):
        assert scrub_text("a cab ran", secrets=["ab"]) == "a cab ran"


def deal(position_id, type, entry, time, volume, price, profit=0.0, commission=0.0, swap=0.0,
         fee=0.0, symbol="XAUUSDm", magic=888992, reason=0, comment=""):
    return dict(position_id=position_id, type=type, entry=entry, time=time, volume=volume,
                price=price, profit=profit, commission=commission, swap=swap, fee=fee,
                symbol=symbol, magic=magic, reason=reason, comment=comment)


DEALS = [
    # 1: short stopped out by the broker. net = -0.07 - 8.59 = -8.66
    deal(1, 1, 0, 1000, 0.01, 3906.75, commission=-0.07),
    deal(1, 0, 1, 1240, 0.01, 3915.34, profit=-8.59, reason=4),
    # 2: long, take profit. net = 12.0 - 0.07 - 0.02 = 11.91
    deal(2, 0, 0, 2000, 0.01, 3900.0, commission=-0.07),
    deal(2, 1, 1, 2600, 0.01, 3912.0, profit=12.0, swap=-0.02, reason=5),
    # 3: 0.02 lot closed in two halves. net = 1.0 + 2.0 = 3.0
    deal(3, 0, 0, 3000, 0.02, 100.0),
    deal(3, 1, 1, 3100, 0.01, 101.0, profit=1.0, reason=3),
    deal(3, 1, 1, 3200, 0.01, 102.0, profit=2.0, reason=5),
    # 4: still open, 5: only half closed, 6: a balance deal, 7: someone else's magic
    deal(4, 0, 0, 4000, 0.01, 50.0),
    deal(5, 0, 0, 5000, 0.02, 50.0),
    deal(5, 1, 1, 5100, 0.01, 51.0, profit=1.0),
    deal(0, 2, 0, 6000, 0.0, 0.0, profit=500.0),
    deal(7, 0, 0, 7000, 0.01, 10.0, magic=0),
    deal(7, 1, 1, 7100, 0.01, 11.0, profit=1.0, magic=0, reason=0),
]


class TestPairDeals:
    def test_only_finished_round_trips_of_the_bots_magic(self):
        trades = pair_deals(DEALS, magic=888992)
        assert [t.position_id for t in trades] == [1, 2, 3]

    def test_without_a_magic_filter_manual_trades_are_included(self):
        assert [t.position_id for t in pair_deals(DEALS)] == [1, 2, 3, 7]

    def test_a_broker_side_stop_out_is_a_loss_with_its_reason_and_costs(self):
        t = pair_deals(DEALS, magic=888992)[0]
        assert t.direction == -1 and t.exit_reason == "sl"
        assert t.net_pnl == pytest.approx(-8.66)
        assert t.duration_s == 240
        assert t.open_price == pytest.approx(3906.75) and t.close_price == pytest.approx(3915.34)

    def test_commission_and_swap_count_in_the_net(self):
        assert pair_deals(DEALS, magic=888992)[1].net_pnl == pytest.approx(11.91)

    def test_partial_closes_are_one_trade_with_a_weighted_exit(self):
        t = pair_deals(DEALS, magic=888992)[2]
        assert t.volume == pytest.approx(0.02)
        assert t.close_price == pytest.approx(101.5)
        assert t.net_pnl == pytest.approx(3.0)
        assert t.exit_reason == "tp"                       # the last exit decides
        assert t.close_time > t.open_time

    def test_empty_history(self):
        assert pair_deals([]) == []


def T(pnl, dur=600, reason="sl", hour=10, direction=1):
    return Trade(position_id=1, symbol="XAUUSDm", direction=direction, volume=0.01,
                 open_time=f"2026-11-02 {hour:02d}:00:00", close_time="2026-11-02 12:00:00",
                 open_price=1.0, close_price=1.0, net_pnl=pnl, exit_reason=reason,
                 duration_s=dur, magic=1, comment="")


class TestTradeStats:
    def setup_method(self):
        self.trades = [T(10, 900, "tp", 9), T(-5, 100, "sl", 9), T(-5, 400, "sl", 10),
                       T(-5, 200, "sl", 10), T(20, 900, "tp", 10)]
        self.s = trade_stats(self.trades)

    def test_headline_numbers(self):
        s = self.s
        assert (s["trades"], s["wins"], s["losses"]) == (5, 2, 3)
        assert s["net"] == 15.0
        assert s["win_rate"] == 0.4
        assert s["avg_win"] == 15.0 and s["avg_loss"] == -5.0
        assert s["profit_factor"] == 2.0                    # 30 / 15
        assert s["expectancy"] == 3.0                       # 15 / 5

    def test_losing_streak_and_quick_losses(self):
        assert self.s["max_consecutive_losses"] == 3
        assert self.s["quick_losses"] == 2                  # the 100 s and 200 s losses

    def test_buckets(self):
        assert self.s["by_exit_reason"]["sl"] == {"trades": 3, "net": -15.0, "win_rate": 0.0}
        assert self.s["by_exit_reason"]["tp"] == {"trades": 2, "net": 30.0, "win_rate": 1.0}
        assert self.s["by_hour"][9]["trades"] == 2 and self.s["by_hour"][10]["trades"] == 3

    def test_median_hold(self):
        assert self.s["median_duration_s"] == 400

    def test_no_trades_gives_none_not_zero(self):
        s = trade_stats([])
        assert s["trades"] == 0 and s["win_rate"] is None and s["profit_factor"] is None

    def test_no_losses_means_profit_factor_is_undefined(self):
        s = trade_stats([T(5), T(7)])
        assert s["profit_factor"] is None and s["avg_loss"] is None and s["win_rate"] == 1.0

    def test_a_breakeven_trade_is_neither_win_nor_loss(self):
        s = trade_stats([T(0.0), T(4.0)])
        assert (s["wins"], s["losses"]) == (1, 0)


LOG = """\
2026-11-01 17:47:51,189 [ERROR] Order failed: retcode=10018 mt5_error=(1, 'Success')
2026-11-02 00:01:31,611 [INFO] [XAUUSDm] Bar 2026-10-30 20:55:00 | direction=+0 score=41.1 ai_conf=0.663 (off-session)
2026-11-02 04:55:01,548 [INFO] [XAUUSDm] Bar 2026-11-02 03:50:00 | direction=+0 score=60.2 ai_conf=0.565 (low ADX (14.7))
2026-11-02 08:00:00,641 [INFO] [XAUUSDm] Bar 2026-11-02 06:55:00 | direction=-1 score=62.8 ai_conf=0.686 (signal)
2026-11-02 08:00:00,652 [INFO] Minimum lot 0.01 risks 8.59 (1.72% of equity) against a 2.89 budget (0.58% configured)
2026-11-02 08:00:00,802 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3906.75100  SL=3915.34300  TP=3889.56700
2026-11-02 08:00:05,024 [INFO] Modified SL ticket=2159773984 → 3909.87200
2026-11-02 08:00:08,024 [INFO] Modified SL ticket=2159773984 → 3909.50000
2026-11-02 08:00:30,024 [INFO] Modified SL ticket=2159773984 → 3909.10000
2026-11-02 08:01:01,024 [INFO] Modified SL ticket=2159773984 → 3908.90000
2026-11-02 08:15:01,248 [INFO] [XAUUSDm] Entry skipped: 1/1 position(s) already open on XAUUSDm
2026-11-02 08:15:15,064 [WARNING] [AI] XAUUSDm challenger rejected (oos accuracy 0.397 below floor 0.520) — keeping live model if one exists
2026-11-02 08:20:01,208 [INFO] Minimum lot 0.01 risks 5.90 (1.18% of equity) against a 1.27 budget (0.25% configured)
2026-11-02 09:00:00,644 [INFO] [XAUUSDm] Bar 2026-11-02 07:55:00 | direction=+0 score=55.4 ai_conf=0.506 (gold AI conf 0.51 < threshold (0.52))
2026-10-29 12:33:39,189 [INFO] 🔒 CLOSE XAUUSDm ticket=2143431219 pnl=+0.24 reason=h1 trend flip
2026-11-02 09:05:00,000 [INFO] [XAUUSDm] Entry blocked: weekly DD 30.0% >= 8.0%
Traceback (most recent call last):
not a log line at all
"""


class TestLogStats:
    def setup_method(self):
        self.log = parse_log(LOG.splitlines(keepends=True))

    def test_window_and_levels(self):
        assert self.log["lines"] == 16
        assert self.log["levels"]["ERROR"] == 1 and self.log["levels"]["WARNING"] == 1

    def test_entries(self):
        (entry,) = self.log["entries"]
        assert entry["side"] == "SELL" and entry["volume"] == 0.01
        assert entry["sl_distance"] == pytest.approx(8.592)
        assert self.log["entries_by_hour"] == {8: 1}

    def test_stop_modifications_per_ticket_and_burst(self):
        sl = self.log["sl_modifications"]
        assert sl["total"] == 4
        assert sl["per_ticket"] == {"2159773984": 4}
        assert sl["max_in_one_minute"] == 3                 # three inside 08:00

    def test_minimum_lot_overshoot(self):
        ml = self.log["min_lot_risk"]
        assert ml["n"] == 2
        assert ml["avg_risk_pct"] == pytest.approx(1.45)
        assert ml["avg_overshoot"] == pytest.approx((8.59 / 2.89 + 5.90 / 1.27) / 2, abs=0.01)
        assert ml["max_overshoot"] == pytest.approx(5.90 / 1.27, abs=0.01)

    def test_order_failures_by_retcode(self):
        assert self.log["order_failed"] == {"10018": 1}

    def test_signal_funnel_groups_numbers_together(self):
        sig = self.log["signals"]
        assert sig["signal"] == 1 and sig["off-session"] == 1
        assert sig["low ADX (#)"] == 1
        assert sig["gold AI conf # < threshold (#)"] == 1

    def test_blocked_and_skipped(self):
        assert self.log["entries_skipped"] == {"#/# position(s) already open on XAUUSDm": 1}
        assert "weekly DD #% >= #%" in self.log["entries_blocked"]

    def test_bot_closes_and_ai_retraining(self):
        assert self.log["bot_closes"] == [{"ticket": "2143431219", "pnl": 0.24, "reason": "h1 trend flip"}]
        assert self.log["ai_challenger_oos_accuracy"] == [0.397]

    def test_empty_log(self):
        log = parse_log([])
        assert log["lines"] == 0 and log["entries"] == [] and log["min_lot_risk"]["n"] == 0


def lines_for(n_mods, ticket="2160013129", minute="09:16"):
    return [f"2026-11-02 {minute}:{i % 60:02d},000 [INFO] Modified SL ticket={ticket} → 3896.{i:03d}\n"
            for i in range(n_mods)]


class TestFindings:
    def test_stop_modification_spam_is_flagged_with_its_numbers(self):
        log = parse_log(lines_for(25))
        found = findings(trade_stats([]), log)
        assert any("25 times on ticket 2160013129" in f for f in found)

    def test_a_few_modifications_are_not_flagged(self):
        assert not any("Stop-loss modified" in f for f in findings(trade_stats([]), parse_log(lines_for(4))))

    def test_minimum_lot_overshoot_is_flagged(self):
        found = findings(trade_stats([]), parse_log(LOG.splitlines(keepends=True)))
        assert any("minimum lot risks on average" in f for f in found)

    def test_losing_results_are_flagged(self):
        stats = trade_stats([T(-5, 100), T(-5, 100), T(-5, 100), T(-5, 100), T(2, 900, "tp")])
        found = findings(stats, parse_log([]))
        assert any("Negative expectancy" in f for f in found)
        assert any("Profit factor" in f for f in found)
        assert any("closed within 5 minutes" in f for f in found)
        assert any("ended at the stop-loss" in f for f in found)
        assert any("losing streak" in f for f in found)

    def test_a_healthy_period_triggers_nothing_but_the_no_trades_note(self):
        found = findings(trade_stats([T(5, 900, "tp"), T(6, 900, "tp")]), parse_log([]))
        assert found == []

    def test_no_trades_is_called_out(self):
        assert any("No closed trades" in f for f in findings(trade_stats([]), parse_log([])))

    def test_market_closed_note_for_retcode_10018(self):
        found = findings(trade_stats([]), parse_log(LOG.splitlines(keepends=True)))
        assert any("10018 = market closed" in f for f in found)

    def test_ai_retraining_below_a_coin_flip(self):
        text = [f"2026-11-02 08:0{i}:00,000 [WARNING] [AI] X challenger rejected "
                f"(oos accuracy 0.3{i} below floor 0.520)\n" for i in range(4)]
        assert any("below a coin flip" in f for f in findings(trade_stats([]), parse_log(text)))

    def test_clustered_same_direction_entries(self):
        text = [f"2026-11-02 08:{m:02d}:00,000 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3900.0  SL=3908.0  TP=3884.0\n"
                for m in (0, 5, 10, 15)]
        assert any("within 15 minutes" in f for f in findings(trade_stats([]), parse_log(text)))


class TestSummary:
    def test_empty_inputs_still_produce_a_summary(self):
        md = build_summary({"generated": "now", "days": 7}, trade_stats([]), parse_log([]))
        assert "# Trading-gold diagnostics" in md and "No closed trades" in md

    def test_summary_carries_the_numbers_and_notes(self):
        stats = trade_stats([T(10, 900, "tp"), T(-5, 100, "sl")])
        md = build_summary({"generated": "now", "days": 7, "account_type": "demo",
                            "account": "***9880", "balance": 500.0, "equity": 498.0},
                           stats, parse_log(LOG.splitlines(keepends=True)),
                           notes=["Only trades with magic number 888992 are included."])
        assert "demo ***9880" in md and "magic number 888992" in md
        assert "## Findings" in md and "By exit reason" in md and "By hour opened" in md
