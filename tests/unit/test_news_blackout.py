"""tests/unit/test_news_blackout.py — scheduled-news entry blackout (pure logic)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.risk.news_blackout import active_event, parse_events

UTC = timezone.utc
NFP = datetime(2026, 11, 6, 13, 30, tzinfo=UTC)


def at(h, m):
    return datetime(2026, 11, 6, h, m, tzinfo=UTC)


class TestParse:
    def test_empty_means_no_blackout(self):
        assert parse_events(None) == ([], [])
        assert parse_events("") == ([], [])
        assert parse_events("  ,  ; ") == ([], [])

    def test_events_are_parsed_as_utc_and_sorted(self):
        ev, bad = parse_events("2026-11-12 13:30, 2026-11-06 13:30;2026-11-06T19:00")
        assert bad == []
        assert ev == [NFP, datetime(2026, 11, 6, 19, 0, tzinfo=UTC), datetime(2026, 11, 12, 13, 30, tzinfo=UTC)]
        assert all(e.tzinfo is UTC for e in ev)

    def test_a_malformed_item_is_reported_not_dropped_silently(self):
        ev, bad = parse_events("2026-11-06 13:30, tomorrow 8pm, 2026-13-45 99:99")
        assert ev == [NFP] and bad == ["tomorrow 8pm", "2026-13-45 99:99"]


class TestWindow:
    def test_edges_are_inclusive_and_outside_is_free(self):
        ev = [NFP]
        assert active_event(at(13, 15), ev, 15, 15) == NFP          # exactly 15 min before
        assert active_event(at(13, 45), ev, 15, 15) == NFP          # exactly 15 min after
        assert active_event(at(13, 14), ev, 15, 15) is None
        assert active_event(at(13, 46), ev, 15, 15) is None

    def test_before_and_after_are_independent(self):
        assert active_event(at(13, 0), [NFP], 30, 0) == NFP
        assert active_event(at(13, 31), [NFP], 30, 0) is None
        assert active_event(at(13, 59), [NFP], 0, 30) == NFP
        assert active_event(at(13, 29), [NFP], 0, 30) is None

    def test_the_matching_event_is_returned_among_several(self):
        later = NFP + timedelta(days=6)
        assert active_event(later, [NFP, later], 5, 5) == later
        assert active_event(NFP + timedelta(days=3), [NFP, later], 5, 5) is None

    def test_no_events_never_blocks_and_negative_minutes_are_clamped(self):
        assert active_event(NFP, [], 15, 15) is None
        assert active_event(at(13, 29), [NFP], -10, -10) is None
        assert active_event(NFP, [NFP], -10, -10) == NFP

    def test_a_naive_clock_is_refused(self):
        with pytest.raises(ValueError):
            active_event(datetime(2026, 11, 6, 13, 30), [NFP], 15, 15)
