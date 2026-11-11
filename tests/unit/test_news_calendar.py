"""The bot's own news calendar: built-in releases, auto-fetched feed, windows, wiring."""

import json
from datetime import datetime, timezone

import pytest

from core.risk.news_calendar import (
    HIGH, MAJOR, NewsCalendar, builtin_events, classify, parse_feed,
)
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot


def at(month, day, hh, mm):
    return datetime(2026, month, day, hh, mm, tzinfo=timezone.utc)


FEED = json.dumps([
    {"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-11-20T08:30:00-05:00",
     "impact": "High"},
    {"title": "Unemployment Claims", "country": "USD", "date": "2026-11-19T08:30:00-05:00",
     "impact": "High"},
    {"title": "ECB Press Conference", "country": "EUR", "date": "2026-11-19T08:45:00-05:00",
     "impact": "High"},
    {"title": "Crude Oil Inventories", "country": "USD", "date": "2026-11-18T10:30:00-05:00",
     "impact": "Medium"},
    {"title": "Broken", "country": "USD", "date": "next friday", "impact": "High"},
    {"title": "Naive", "country": "USD", "date": "2026-11-19T08:30:00", "impact": "High"},
])


# ── classification & built-in list ─────────────────────────────────────────

@pytest.mark.parametrize("title,tier", [
    ("Non-Farm Employment Change", MAJOR), ("Non-Farm Payrolls (November)", MAJOR),
    ("CPI m/m", MAJOR), ("Core CPI (November)", MAJOR),
    ("FOMC Federal Funds Rate decision", MAJOR), ("FOMC Press Conference", MAJOR),
    ("PPI (October)", HIGH), ("Retail Sales (October)", HIGH), ("JOLTS Job Openings", HIGH),
])
def test_classify(title, tier):
    assert classify(title) == tier


def test_builtin_has_the_big_december_releases_in_utc():
    by_time = {e.when: e for e in builtin_events()}
    assert by_time[at(12, 4, 13, 30)].tier == MAJOR            # NFP 8:30 ET
    assert by_time[at(12, 10, 13, 30)].tier == MAJOR           # CPI 8:30 ET
    assert by_time[at(12, 9, 19, 0)].tier == MAJOR             # FOMC 14:00 ET
    assert all(e.when.tzinfo is not None for e in by_time.values())


# ── feed parsing ───────────────────────────────────────────────────────────

def test_feed_keeps_only_high_impact_in_the_wanted_currency():
    events = parse_feed(FEED)
    assert [(e.title, e.when) for e in events] == [
        ("Non-Farm Employment Change", at(11, 20, 13, 30)),
        ("Unemployment Claims", at(11, 19, 13, 30)),
    ]
    assert events[0].tier == MAJOR and events[1].tier == HIGH


def test_feed_never_guesses_a_timezone_and_skips_bad_dates():
    assert "Naive" not in [e.title for e in parse_feed(FEED)]
    assert "Broken" not in [e.title for e in parse_feed(FEED)]


def test_feed_other_currencies_on_request():
    titles = [e.title for e in parse_feed(FEED, currencies=("USD", "EUR"))]
    assert "ECB Press Conference" in titles


def test_a_block_page_is_an_error_not_an_empty_calendar():
    with pytest.raises(ValueError):
        parse_feed("<html>Web Page Blocked!</html>")
    with pytest.raises(ValueError):
        parse_feed('{"error": "rate limited"}')


# ── windows ────────────────────────────────────────────────────────────────

def cal(**kw):
    return NewsCalendar(cache_path=None, auto_fetch=False, background=False, **kw)


def test_major_events_get_the_longer_window():
    c = cal()
    nfp = at(12, 4, 13, 30)
    kw = dict(before_min=20, after_min=20, major_before_min=30, major_after_min=60)
    assert c.active(at(12, 4, 13, 0), **kw) is not None           # 30 min before
    assert c.active(at(12, 4, 14, 30), **kw) is not None          # 60 min after
    assert c.active(at(12, 4, 14, 31), **kw) is None
    assert c.active(at(12, 4, 12, 59), **kw) is None
    assert c.active(nfp, **kw).title.startswith("Non-Farm")


def test_ordinary_high_events_use_the_normal_window():
    c = cal()
    kw = dict(before_min=20, after_min=20, major_before_min=30, major_after_min=60)
    assert c.active(at(11, 17, 13, 50), **kw) is not None         # retail sales +20
    assert c.active(at(11, 17, 13, 51), **kw) is None


def test_user_events_still_apply_and_builtin_can_be_turned_off():
    c = cal(use_builtin=False)
    assert c.active(at(12, 4, 13, 30)) is None
    assert c.active(at(11, 6, 13, 30), user_events=[at(11, 6, 13, 30)]).source == "user"


def test_naive_time_is_refused():
    with pytest.raises(ValueError):
        cal().active(datetime(2026, 12, 4, 13, 30))


# ── auto-fetch ─────────────────────────────────────────────────────────────

def test_fetch_adds_events_and_writes_a_cache_that_survives_a_restart(tmp_path):
    calls = []
    cache = tmp_path / "news.json"
    c = NewsCalendar(cache_path=str(cache), auto_fetch=True, background=False,
                     use_builtin=False, fetch=lambda url: calls.append(url) or FEED)
    c.maybe_refresh(at(11, 16, 0, 0))
    assert len(calls) == 1 and "fetched 2" in c.last_status
    assert c.active(at(11, 20, 13, 30)).source == "feed"
    restarted = NewsCalendar(cache_path=str(cache), auto_fetch=False, use_builtin=False)
    assert restarted.active(at(11, 20, 13, 30)) is not None


def test_fetch_is_rate_limited(tmp_path):
    calls = []
    c = NewsCalendar(cache_path=None, auto_fetch=True, background=False, refresh_hours=6,
                     fetch=lambda url: calls.append(url) or FEED)
    c.maybe_refresh(at(11, 16, 0, 0))
    c.maybe_refresh(at(11, 16, 5, 59))
    c.maybe_refresh(at(11, 16, 6, 0))
    assert len(calls) == 2


def test_a_failed_fetch_keeps_the_builtin_list_and_the_last_good_feed(tmp_path):
    good = NewsCalendar(cache_path=None, auto_fetch=True, background=False,
                        fetch=lambda url: FEED)
    good.maybe_refresh(at(11, 16, 0, 0))

    def boom(url):
        raise OSError("blocked")
    good._fetch = boom
    good.maybe_refresh(at(11, 17, 0, 0))
    assert "fetch failed" in good.last_status
    assert good.active(at(11, 20, 13, 30)) is not None          # last good feed kept
    assert good.active(at(12, 4, 13, 30)) is not None           # built-in kept


def test_auto_fetch_off_never_calls_the_network():
    c = NewsCalendar(cache_path=None, auto_fetch=False, background=False,
                     fetch=lambda url: pytest.fail("network must not be called"))
    c.maybe_refresh(at(11, 16, 0, 0))


# ── wiring through run.py ──────────────────────────────────────────────────

def test_bot_blocks_entries_around_nfp_by_itself(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"), news_blackout="",
                   news_builtin=True, news_auto_fetch=False)
    reason = bot._news_blackout_reason(at(12, 4, 13, 10))
    assert "news blackout around 2026-12-04 13:30 UTC" in reason and "Non-Farm" in reason
    assert bot._news_blackout_reason(at(12, 4, 15, 0)) == ""


def test_builtin_off_restores_the_user_only_behaviour(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"), news_blackout="",
                   news_builtin=False, news_auto_fetch=False)
    assert bot._news_blackout_reason(at(12, 4, 13, 30)) == ""


def test_defaults_builtin_on_download_off(run_and_stub):
    run_module, _ = run_and_stub
    from core.config.schema import PARAMS
    assert run_module.CONFIG["news_builtin"] is True
    assert run_module.CONFIG["news_auto_fetch"] is False
    assert PARAMS["news_auto_fetch"].default is False
    assert PARAMS["news_major_after_min"].default == 60.0
