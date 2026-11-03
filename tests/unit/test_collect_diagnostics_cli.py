"""
tests/unit/test_collect_diagnostics_cli.py — scripts/collect_diagnostics.py end
to end, with a fake MT5 that is strictly read-only, secrets planted in .env and
in the log, and a check that none of them reaches any file in the zip.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import zipfile
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PASSWORD, TOKEN, LOGIN = "Sup3rSecret!", "tok_9f8e7d6c5b", "261009880"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

Deal = namedtuple("Deal", "ticket order position_id type entry time volume price profit commission "
                          "swap fee symbol magic reason comment")


def deals():
    def d(pos, typ, entry, t, vol, price, profit=0.0, magic=888992, reason=0, commission=0.0):
        return Deal(1, 1, pos, typ, entry, t, vol, price, profit, commission, 0.0, 0.0,
                    "XAUUSDm", magic, reason, "")
    return (
        d(1, 1, 0, 1000, 0.01, 3906.75, commission=-0.07), d(1, 0, 1, 1240, 0.01, 3915.34, -8.59, reason=4),
        d(2, 0, 0, 2000, 0.01, 3900.0), d(2, 1, 1, 2600, 0.01, 3912.0, 12.0, reason=5),
        d(3, 0, 0, 3000, 0.01, 10.0, magic=0), d(3, 1, 1, 3100, 0.01, 11.0, 1.0, magic=0),
    )


class FakeMT5:
    """Read-only. Touching any order function fails the test."""

    def __init__(self, ok=True):
        self.ok, self.closed, self.range = ok, False, None

    def initialize(self):
        return self.ok

    def last_error(self):
        return (-10005, "IPC timeout")

    def account_info(self):
        return SimpleNamespace(trade_mode=0, login=int(LOGIN), server="Exness-MT5Trial16",
                               currency="USD", leverage=200, balance=500.0, equity=498.7)

    def history_deals_get(self, start, end):
        self.range = (start, end)
        return deals()

    def shutdown(self):
        self.closed = True

    def __getattr__(self, name):
        raise AssertionError(f"the collector must be read-only, but it touched mt5.{name}")


@pytest.fixture(scope="module")
def cli():
    path = REPO_ROOT / "scripts" / "collect_diagnostics.py"
    spec = importlib.util.spec_from_file_location("collect_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["collect_diagnostics"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot_dir(tmp_path):
    (tmp_path / ".env").write_text(
        f"MT5_LOGIN={LOGIN}\nMT5_PASSWORD={PASSWORD}\nDASHBOARD_TOKEN={TOKEN}\n"
        "MT5_SERVER=Exness-MT5Trial16\nMAGIC_NUMBER=888992\nRISK_PER_TRADE=0.01\n"
        "DATABASE_URL=sqlite:///./data/trading.db\nRISK_STATE_DIR=data/state\n", encoding="utf-8")
    (tmp_path / "trading_bot.log").write_text(
        f"2026-11-02 08:00:00,000 [INFO] Already logged in to terminal account {LOGIN}. Bypassing login step.\n"
        f"2026-11-02 08:00:01,000 [INFO] retrying with password={PASSWORD} header X-Dashboard-Token: {TOKEN}\n"
        "2026-11-02 08:00:02,802 [INFO] ✅ SELL XAUUSDm vol=0.01 @ 3906.75100  SL=3915.34300  TP=3889.56700\n"
        "2026-11-02 08:00:05,024 [INFO] Modified SL ticket=2159773984 → 3909.87200\n", encoding="utf-8")
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    (state / "risk_XAUUSDm.json").write_text(json.dumps({"daily_start_equity": 500.0}), encoding="utf-8")
    models = tmp_path / "data" / "models"
    models.mkdir()
    (models / "nn_XAUUSDm_M5.pt.history.jsonl").write_text('{"action": "kept"}\n', encoding="utf-8")
    conn = sqlite3.connect(tmp_path / "data" / "trading.db")
    conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, pnl REAL, "
                 "balance REAL, exit_reason TEXT)")
    conn.execute("INSERT INTO trades (timestamp, symbol, pnl, balance, exit_reason) "
                 "VALUES ('2026-11-02T08:30:00', 'XAUUSDm', 0.24, 500.0, 'h1 trend flip')")
    conn.commit()
    conn.close()
    return tmp_path


def members(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return {name: zf.read(name).decode("utf-8") for name in zf.namelist()}


def run(cli, bot_dir, mt5=None, **kw):
    return cli.collect(bot_dir, bot_dir / "diagnostics", days=7, mt5=mt5 or FakeMT5(), now=NOW, **kw)


class TestBundle:
    def test_contains_the_expected_files(self, cli, bot_dir):
        names = set(members(run(cli, bot_dir)))
        assert {"summary.md", "stats.json", "closed_trades.csv", "log_tail.txt",
                "config_redacted.env", "account.json", "trades_db.csv",
                "risk_state/risk_XAUUSDm.json", "nn/nn_XAUUSDm_M5.pt.history.jsonl"} <= names

    def test_no_secret_reaches_any_file_in_the_zip(self, cli, bot_dir):
        for name, text in members(run(cli, bot_dir)).items():
            for secret in (PASSWORD, TOKEN, LOGIN):
                assert secret not in text, f"{secret!r} leaked into {name}"

    def test_the_log_keeps_its_useful_content_after_scrubbing(self, cli, bot_dir):
        log = members(run(cli, bot_dir))["log_tail.txt"]
        assert "✅ SELL XAUUSDm" in log and "Modified SL ticket=2159773984" in log
        assert "***9880" in log

    def test_the_account_is_reported_as_demo_with_a_masked_login(self, cli, bot_dir):
        account = json.loads(members(run(cli, bot_dir))["account.json"])
        assert account["type"] == "demo" and account["login"] == "***9880"
        assert account["balance"] == 500.0

    def test_only_the_bots_magic_number_by_default(self, cli, bot_dir):
        rows = members(run(cli, bot_dir))["closed_trades.csv"].strip().splitlines()
        assert len(rows) == 1 + 2                              # header + positions 1 and 2

    def test_all_magics_includes_manual_trades(self, cli, bot_dir):
        rows = members(run(cli, bot_dir, use_env_magic=False))["closed_trades.csv"].strip().splitlines()
        assert len(rows) == 1 + 3

    def test_summary_carries_the_real_results(self, cli, bot_dir):
        md = members(run(cli, bot_dir))["summary.md"]
        assert "demo ***9880" in md
        assert "| 2 | 3.34 |" in md                            # two trades, net -8.66 + 12.0 = 3.34
        assert "magic number 888992" in md

    def test_the_trade_database_is_exported(self, cli, bot_dir):
        assert "h1 trend flip" in members(run(cli, bot_dir))["trades_db.csv"]

    def test_the_history_window_covers_the_requested_days(self, cli, bot_dir):
        fake = FakeMT5()
        run(cli, bot_dir, mt5=fake)
        start, end = fake.range
        assert (NOW - start).days == 7 and end > NOW

    def test_mt5_is_shut_down_and_never_asked_to_trade(self, cli, bot_dir):
        fake = FakeMT5()
        run(cli, bot_dir, mt5=fake)                            # an order call would raise
        assert fake.closed is True

    def test_the_bot_files_are_not_modified(self, cli, bot_dir):
        before = {p: p.read_bytes() for p in bot_dir.rglob("*") if p.is_file()}
        run(cli, bot_dir)
        for path, data in before.items():
            assert path.read_bytes() == data, f"{path} changed"

    def test_latest_summary_is_written_next_to_the_zip(self, cli, bot_dir):
        run(cli, bot_dir)
        assert (bot_dir / "diagnostics" / "latest_summary.md").exists()


class TestDegradedCases:
    def test_no_mt5_still_produces_a_bundle_with_a_note(self, cli, bot_dir, monkeypatch):
        monkeypatch.setitem(sys.modules, "MetaTrader5", None)           # import -> ImportError
        path = cli.collect(bot_dir, bot_dir / "diagnostics", now=NOW, mt5=None)
        md = members(path)["summary.md"]
        assert "MetaTrader5 is not available" in md and "No closed trades" in md
        assert "✅ SELL" in members(path)["log_tail.txt"]

    def test_an_mt5_that_will_not_initialise_is_reported_not_fatal(self, cli, bot_dir):
        md = members(run(cli, bot_dir, mt5=FakeMT5(ok=False)))["summary.md"]
        assert "MT5 initialize failed" in md

    def test_missing_env_and_log_still_work(self, cli, tmp_path):
        path = cli.collect(tmp_path, tmp_path / "out", now=NOW, mt5=FakeMT5())
        md = members(path)["summary.md"]
        assert "No trading_bot.log found" in md

    def test_it_refuses_to_write_a_bundle_if_a_secret_would_survive(self, cli, bot_dir, monkeypatch):
        monkeypatch.setattr(cli, "scrub_text", lambda text, *a, **k: text)    # a broken scrubber
        monkeypatch.setattr(cli, "redact_env_text", lambda text: text)
        with pytest.raises(RuntimeError, match="refusing to write"):
            run(cli, bot_dir)
        assert not list((bot_dir / "diagnostics").glob("*.zip"))

    def test_main_reports_the_refusal_with_a_failing_exit_code(self, cli, bot_dir, monkeypatch, capsys):
        monkeypatch.setattr(cli, "scrub_text", lambda text, *a, **k: text)
        monkeypatch.setattr(cli, "redact_env_text", lambda text: text)
        assert cli.main(["--root", str(bot_dir)], mt5=FakeMT5()) == 1
        assert "refusing to write" in capsys.readouterr().out

    def test_main_prints_where_to_look(self, cli, bot_dir, capsys):
        assert cli.main(["--root", str(bot_dir), "--days", "3"], mt5=FakeMT5()) == 0
        out = capsys.readouterr().out
        assert "wrote" in out and "latest_summary.md" in out


class TestCompare:
    def test_no_compare_section_unless_asked(self, cli, bot_dir):
        files = members(run(cli, bot_dir))
        assert "A/B comparison" not in files["summary.md"]
        assert json.loads(files["stats.json"])["compare"] is None

    def test_compare_uses_all_magics_even_though_the_default_filter_hides_them(self, cli, bot_dir):
        files = members(run(cli, bot_dir, compare=[888992, 0]))
        ab = json.loads(files["stats.json"])["compare"]
        assert ab["baseline"] == "888992"
        assert ab["variants"]["888992"]["trades"] == 2 and ab["variants"]["0"]["trades"] == 1
        assert ab["variants"]["888992"]["net"] == pytest.approx(3.34, abs=0.005)    # -8.59 - 0.07 + 12.0
        assert "A/B comparison" in files["summary.md"] and "Too few trades" in files["summary.md"]

    def test_main_rejects_a_bad_compare_list(self, cli, bot_dir, capsys):
        for bad in ("888992", "888992,888992", "a,b"):
            assert cli.main(["--root", str(bot_dir), "--compare", bad], mt5=FakeMT5()) == 2
        assert "error" in capsys.readouterr().out

    def test_main_accepts_a_compare_list(self, cli, bot_dir, capsys):
        assert cli.main(["--root", str(bot_dir), "--compare", "888992,0"], mt5=FakeMT5()) == 0
