"""
tests/unit/test_diagnostics_origin.py — who opened the trades?

Real case this guards: an --all-magics report showed -256.63 over 113 trades,
while the bot's own 79 trades had lost -35.04. The other 34 were not the bot's
(different magic number) and 11 were closed by hand from the MT5 mobile app and
terminal. The report must say that outright instead of blending them.
"""

from __future__ import annotations

import json

import pytest

from core.diagnostics.logstats import parse_log
from core.diagnostics.report import build_summary, findings
from core.diagnostics.trades import HAND_CLOSED, Trade, trade_stats
from tests.unit.test_collect_diagnostics_cli import (  # noqa: F401
    FakeMT5, NOW, bot_dir, cli, members, run,
)

BOT = 888992


def trade(pnl, magic=BOT, reason="sl", position_id=1):
    return Trade(position_id=position_id, symbol="XAUUSDm", direction=1, volume=0.01,
                 open_time="2026-11-03 11:00:00", close_time="2026-11-03 11:30:00",
                 open_price=1.0, close_price=1.0, net_pnl=float(pnl), exit_reason=reason,
                 duration_s=1800, magic=magic, comment="")


# bot: +10 sl, -5 sl, +2 expert.   other: -30 mobile, -20 mobile, -8 client, -4 sl
TRADES = [trade(10), trade(-5), trade(2, reason="expert"),
          trade(-30, magic=0, reason="mobile"), trade(-20, magic=0, reason="mobile"),
          trade(-8, magic=0, reason="client"), trade(-4, magic=7, reason="sl")]


class TestStats:
    def setup_method(self):
        self.s = trade_stats(TRADES, bot_magic=BOT)

    def test_origin_split_hand_computed(self):
        o = self.s["by_origin"]
        assert o["bot"] == {"trades": 3, "net": 7.0, "win_rate": pytest.approx(0.667, abs=0.001)}
        assert o["other"] == {"trades": 4, "net": -62.0, "win_rate": 0.0}

    def test_a_different_magic_number_of_any_value_is_other(self):
        assert trade_stats([trade(1, magic=7)], bot_magic=BOT)["by_origin"]["other"]["trades"] == 1

    def test_trades_closed_by_hand_are_counted_with_their_money(self):
        assert self.s["hand_closed"] == {"trades": 3, "net": -58.0}
        assert set(HAND_CLOSED) == {"client", "mobile", "web"}

    def test_a_stop_or_the_bot_closing_a_trade_is_not_by_hand(self):
        s = trade_stats([trade(-5, reason="sl"), trade(3, reason="expert"), trade(1, reason="tp")],
                        bot_magic=BOT)
        assert s["hand_closed"] == {"trades": 0, "net": 0.0}

    def test_no_bot_magic_means_no_origin_split(self):
        assert trade_stats(TRADES)["by_origin"] == {}

    def test_empty_history_keeps_the_new_keys(self):
        s = trade_stats([], bot_magic=BOT)
        assert s["by_origin"] == {} and s["hand_closed"] == {"trades": 0, "net": 0.0}

    def test_the_headline_numbers_are_unchanged_by_the_split(self):
        assert self.s["trades"] == 7 and self.s["net"] == -55.0


class TestFinding:
    def test_the_finding_states_who_lost_the_money_with_numbers(self):
        found = findings(trade_stats(TRADES, bot_magic=BOT), parse_log([]))
        first = found[0]
        assert "4 of 7 trades (net -62.0) were NOT opened by the bot" in first
        assert "3 trades (net -58.0) were closed by hand" in first
        assert "The bot's own 3 trades: net 7.0" in first

    def test_it_comes_first_because_it_changes_how_everything_else_reads(self):
        assert "NOT opened by the bot" in findings(trade_stats(TRADES, bot_magic=BOT), parse_log([]))[0]

    def test_no_finding_when_every_trade_is_the_bots(self):
        stats = trade_stats([trade(5), trade(-2)], bot_magic=BOT)
        assert not any("NOT opened" in f for f in findings(stats, parse_log([])))

    def test_no_finding_without_the_bot_magic(self):
        assert not any("NOT opened" in f for f in findings(trade_stats(TRADES), parse_log([])))

    def test_the_summary_has_an_origin_table(self):
        md = build_summary({"generated": "now", "days": 7}, trade_stats(TRADES, bot_magic=BOT),
                           parse_log([]))
        assert "By origin" in md and "| bot | 3 | 7.0 |" in md and "| other | 4 | -62.0 |" in md


class TestCollector:
    def test_all_magics_report_splits_bot_from_manual(self, cli, bot_dir):
        data = json.loads(members(run(cli, bot_dir, use_env_magic=False))["stats.json"])
        origin = data["trades"]["by_origin"]
        assert origin["bot"]["trades"] == 2 and origin["other"]["trades"] == 1
        # the fixture's magic-0 position was closed with reason 0 = "client" (by hand)
        assert data["trades"]["hand_closed"]["trades"] == 1

    def test_all_magics_summary_says_so(self, cli, bot_dir):
        md = members(run(cli, bot_dir, use_env_magic=False))["summary.md"]
        assert "NOT opened by the bot" in md
        assert "All magic numbers are included" in md

    def test_the_default_report_is_the_bots_trades_only(self, cli, bot_dir):
        data = json.loads(members(run(cli, bot_dir))["stats.json"])
        assert data["trades"]["by_origin"].get("other") is None
        assert data["trades"]["trades"] == 2
        assert "Only trades with magic number 888992" in members(run(cli, bot_dir))["summary.md"]
