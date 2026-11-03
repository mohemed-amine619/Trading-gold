"""
tests/unit/test_news_and_notifier_wiring.py — through run.py itself:

* the news blackout stops NEW entries (the gate), reports typos loudly, and never touches
  position management;
* the Telegram notifier is off without credentials, never lets the token reach a log line,
  drops identical repeats, and never raises into the trading loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot

NFP = datetime(2026, 11, 6, 13, 30, tzinfo=timezone.utc)


def at(h, m):
    return datetime(2026, 11, 6, h, m, tzinfo=timezone.utc)


class TestNewsBlackoutWiring:
    def _bot(self, run_module, tmp_path, **cfg):
        bot = make_bot(run_module, str(tmp_path / "r.json"), **cfg)
        bot.daily_start = 389.0
        return bot

    def test_off_by_default(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["news_blackout"] == ""
        assert self._bot(run_module, tmp_path)._news_blackout_reason(NFP) == ""

    def test_blocks_inside_the_window_and_allows_outside(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, news_blackout="2026-11-06 13:30",
                        news_blackout_before_min=15, news_blackout_after_min=15)
        assert "news blackout around 2026-11-06 13:30 UTC" in bot._news_blackout_reason(at(13, 20))
        assert bot._news_blackout_reason(at(13, 14)) == ""
        assert bot._news_blackout_reason(at(13, 46)) == ""

    def test_the_real_gate_refuses_a_new_entry_during_the_window(self, run_and_stub, tmp_path, monkeypatch):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, news_blackout="2026-11-06 13:30")
        monkeypatch.setattr(bot, "_news_blackout_reason", lambda now=None: bot.__class__._news_blackout_reason(bot, NFP))
        allowed, mult, reason = bot._entry_gate(389.0)
        assert (allowed, mult) == (False, 0.0) and "news blackout" in reason

    def test_a_typo_is_logged_once_and_the_valid_events_still_apply(self, run_and_stub, tmp_path, caplog):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, news_blackout="2026-11-06 13:30, friday 8:30")
        with caplog.at_level(logging.ERROR):
            assert bot._news_blackout_reason(NFP) != ""
            bot._news_blackout_reason(NFP)
        errors = [r for r in caplog.records if "NEWS_BLACKOUT" in r.getMessage()]
        assert len(errors) == 1 and "friday 8:30" in errors[0].getMessage()

    def test_editing_the_setting_takes_effect_without_a_restart(self, run_and_stub, tmp_path):
        run_module, _ = run_and_stub
        bot = self._bot(run_module, tmp_path, news_blackout="2026-11-06 13:30")
        assert bot._news_blackout_reason(NFP) != ""
        bot.cfg["news_blackout"] = "2026-12-04 13:30"
        assert bot._news_blackout_reason(NFP) == ""

    def test_the_settings_are_declared(self):
        from core.config.schema import PARAMS
        assert PARAMS["news_blackout"].env == "NEWS_BLACKOUT" and PARAMS["news_blackout"].default == ""
        assert PARAMS["news_blackout_before_min"].default == 15.0


class FakeResponse:
    def __init__(self, status=200):
        self.status_code = status


class TestNotifier:
    TOKEN = "123456:SECRET-TOKEN-VALUE"

    def _notifier(self, run_module, post, monkeypatch, token=TOKEN, chat="42"):
        monkeypatch.setattr(run_module.time, "sleep", lambda s: None)
        return run_module.Notifier({"telegram_token": token, "telegram_chat_id": chat}, post=post)

    def test_without_credentials_nothing_is_sent(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        calls = []
        n = self._notifier(run_module, lambda *a, **k: calls.append(1), monkeypatch, token="", chat="")
        n._send_telegram("hello")
        assert calls == [] and n.telegram_enabled is False

    def test_the_token_is_not_in_config_so_the_api_cannot_serve_it(self, run_and_stub):
        run_module, _ = run_and_stub
        assert "telegram_token" not in run_module.CONFIG and "telegram_chat_id" not in run_module.CONFIG

    def test_a_message_is_posted_to_the_chat(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        seen = []
        n = self._notifier(run_module, lambda url, **k: seen.append((url, k)) or FakeResponse(), monkeypatch)
        n._send_telegram("🤖 started")
        url, kw = seen[0]
        assert url == f"https://api.telegram.org/bot{self.TOKEN}/sendMessage"
        assert kw["data"] == {"chat_id": "42", "text": "🤖 started"} and kw["timeout"] == (5, 10)

    def test_an_identical_message_inside_the_window_is_dropped(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        seen = []
        n = self._notifier(run_module, lambda *a, **k: seen.append(1) or FakeResponse(), monkeypatch)
        n._send_telegram("same")
        n._send_telegram("same")
        n._send_telegram("different")
        assert len(seen) == 2

    def test_a_failing_transport_never_raises_and_never_logs_the_token(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub

        def boom(url, **k):
            raise ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")
        n = self._notifier(run_module, boom, monkeypatch)
        with caplog.at_level(logging.DEBUG):
            n._send_telegram("hello")
            n.send("via the thread path")          # the public non-blocking entry must not raise either
        assert self.TOKEN not in caplog.text and "SECRET-TOKEN-VALUE" not in caplog.text
        assert "ConnectionError" in caplog.text

    def test_http_errors_back_off_instead_of_hammering(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        calls = []
        n = self._notifier(run_module, lambda *a, **k: calls.append(1) or FakeResponse(500), monkeypatch)
        n._send_telegram("one")
        first = len(calls)
        n._send_telegram("two")                    # inside the 60 s cooldown: skipped quietly
        assert first == 3 and len(calls) == 3
