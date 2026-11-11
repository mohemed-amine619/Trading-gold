"""
core/risk/news_calendar.py — the bot knows the news calendar by itself.

Three sources are merged, and any one may be missing without breaking the others:

1. a BUILT-IN list of confirmed high-impact US releases (works offline);
2. an optional AUTO-FETCHED weekly feed (Forex Factory's public JSON export),
   filtered to the configured currencies and to high impact, cached on disk so
   a restart or an outage keeps the last good copy;
3. the user's own NEWS_BLACKOUT times (``core/risk/news_blackout.parse_events``).

Two window sizes: "major" events (NFP, CPI, FOMC decision / press conference)
move gold the most and keep spreads wide for longer, so they get their own
before/after minutes; every other high-impact event uses the normal window.

Safety rules:
* The blackout only blocks NEW entries; it never closes or loosens anything.
* Fetching never runs in the trading loop: it is started on a daemon thread at
  most every ``refresh_hours`` and the loop only reads what is already there.
* A failed or malformed download is logged and ignored; the built-in list and
  the user's list still apply. Nothing here can make the bot trade MORE.
* The request carries no user data: a plain GET with a generic User-Agent.

No broker import; Linux-testable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
USER_AGENT = "Mozilla/5.0 (compatible; trading-bot news calendar)"

MAJOR = "major"
HIGH = "high"

_MAJOR_PATTERNS = re.compile(
    r"non-?farm|\bNFP\b|employment situation|\bCPI\b|consumer price|"
    r"\bFOMC\b|federal funds|fed(eral)? (interest )?rate|powell|fed chair",
    re.IGNORECASE)


@dataclass(frozen=True)
class NewsEvent:
    when: datetime          # timezone-aware, UTC
    title: str
    tier: str               # MAJOR or HIGH
    source: str             # builtin / feed / user


def classify(title: str) -> str:
    """MAJOR for NFP, CPI, FOMC; HIGH for everything else."""
    return MAJOR if _MAJOR_PATTERNS.search(title or "") else HIGH


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


# Confirmed high-impact US releases (UTC). EST (UTC-5) applies until 2027-03-14,
# so 8:30 ET = 13:30 UTC, 10:00 ET = 15:00 UTC, 14:00 ET = 19:00 UTC.
# Sources: BLS release schedules, Federal Reserve FOMC calendar. Add new dates as
# the agencies publish them; past dates are harmless.
_BUILTIN = (
    ("2026-11-13 13:30", "PPI (October)"),
    ("2026-11-17 13:30", "Retail Sales (October)"),
    ("2026-11-18 19:00", "FOMC Meeting Minutes"),
    ("2026-11-25 13:30", "GDP Q3 second estimate / PCE"),
    ("2026-12-01 15:00", "JOLTS Job Openings"),
    ("2026-12-04 13:30", "Non-Farm Payrolls (November)"),
    ("2026-12-09 19:00", "FOMC Federal Funds Rate decision"),
    ("2026-12-09 19:30", "FOMC Press Conference"),
    ("2026-12-10 13:30", "CPI (November)"),
    ("2026-12-15 13:30", "PPI (November)"),
    ("2026-12-16 13:30", "Retail Sales (November)"),
    ("2026-12-23 13:30", "GDP Q3 third estimate / PCE"),
)


def builtin_events() -> List[NewsEvent]:
    return [NewsEvent(_utc(t), name, classify(name), "builtin") for t, name in _BUILTIN]


def parse_feed(text: str, currencies: Sequence[str] = ("USD",),
               impacts: Sequence[str] = ("High",)) -> List[NewsEvent]:
    """Parse the weekly JSON export: a list of objects with ``title``,
    ``country`` (currency code), ``date`` (ISO 8601 with offset) and ``impact``.

    Items with an unreadable date or another currency/impact are skipped. Raises
    ``ValueError`` when the document itself is not a JSON list, so a block page or
    an HTML error is reported instead of silently meaning "no news".
    """
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("news feed is not a JSON list")
    want_ccy = {c.strip().upper() for c in currencies if c.strip()}
    want_imp = {i.strip().lower() for i in impacts if i.strip()}
    out: List[NewsEvent] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if str(item.get("country", "")).upper() not in want_ccy:
            continue
        if str(item.get("impact", "")).lower() not in want_imp:
            continue
        try:
            when = datetime.fromisoformat(str(item["date"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if when.tzinfo is None:
            continue                                  # never guess a timezone
        title = str(item.get("title", "")).strip() or "news"
        out.append(NewsEvent(when.astimezone(timezone.utc), title, classify(title), "feed"))
    return out


def _http_get(url: str, timeout_s: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:     # noqa: S310 (fixed https URL)
        return resp.read().decode("utf-8", errors="replace")


class NewsCalendar:
    """Merged calendar with a background, cached auto-fetch."""

    def __init__(self, cache_path: Optional[str] = None, feed_url: str = DEFAULT_FEED_URL,
                 currencies: Sequence[str] = ("USD",), auto_fetch: bool = True,
                 use_builtin: bool = True, refresh_hours: float = 6.0,
                 fetch: Optional[Callable[[str], str]] = None,
                 background: bool = True):
        self.cache_path = cache_path
        self.feed_url = feed_url
        self.currencies = tuple(currencies)
        self.auto_fetch = auto_fetch
        self.use_builtin = use_builtin
        self.refresh = timedelta(hours=max(0.25, float(refresh_hours)))
        self._fetch = fetch or _http_get
        self._background = background
        self._lock = threading.Lock()
        self._feed: List[NewsEvent] = []
        self._last_attempt: Optional[datetime] = None
        self._running = False
        self.last_status = "not fetched"
        self._load_cache()

    # -- cache -----------------------------------------------------------
    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, encoding="utf-8") as fh:
                self._feed = parse_feed(fh.read(), self.currencies)
            self.last_status = f"loaded {len(self._feed)} event(s) from cache"
        except Exception as exc:
            logger.warning("news cache unreadable, ignored: %r", exc)

    def _save_cache(self, text: str) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            logger.warning("could not write news cache: %r", exc)

    # -- fetching --------------------------------------------------------
    def _do_fetch(self) -> None:
        try:
            text = self._fetch(self.feed_url)
            events = parse_feed(text, self.currencies)
            with self._lock:
                self._feed = events
            self._save_cache(text)
            self.last_status = f"fetched {len(events)} high-impact event(s)"
            logger.info("News calendar: %s from %s", self.last_status, self.feed_url)
        except Exception as exc:
            self.last_status = f"fetch failed: {exc!r}"
            logger.warning("News calendar fetch failed (built-in and NEWS_BLACKOUT still "
                           "apply): %r", exc)
        finally:
            self._running = False

    def maybe_refresh(self, now: datetime) -> None:
        """Start a fetch if one is due. Never blocks when ``background`` is on."""
        if not self.auto_fetch or self._running:
            return
        if self._last_attempt is not None and now - self._last_attempt < self.refresh:
            return
        self._last_attempt = now
        self._running = True
        if self._background:
            threading.Thread(target=self._do_fetch, name="news-calendar", daemon=True).start()
        else:
            self._do_fetch()

    # -- reading ---------------------------------------------------------
    def events(self, user_events: Iterable[datetime] = ()) -> List[NewsEvent]:
        out: List[NewsEvent] = builtin_events() if self.use_builtin else []
        with self._lock:
            out.extend(self._feed)
        out.extend(NewsEvent(t, "NEWS_BLACKOUT", HIGH, "user") for t in user_events)
        return sorted(out, key=lambda e: e.when)

    def active(self, now: datetime, user_events: Iterable[datetime] = (),
               before_min: float = 20.0, after_min: float = 20.0,
               major_before_min: float = 30.0, major_after_min: float = 60.0
               ) -> Optional[NewsEvent]:
        """The event whose window contains ``now`` (edges inclusive), else None."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware (UTC)")
        for ev in self.events(user_events):
            major = ev.tier == MAJOR
            before = timedelta(minutes=max(0.0, major_before_min if major else before_min))
            after = timedelta(minutes=max(0.0, major_after_min if major else after_min))
            if ev.when - before <= now <= ev.when + after:
                return ev
        return None

    def upcoming(self, now: datetime, hours: float = 48.0,
                 user_events: Iterable[datetime] = ()) -> List[NewsEvent]:
        end = now + timedelta(hours=hours)
        return [e for e in self.events(user_events) if now <= e.when <= end]


_shared: dict = {}
_shared_lock = threading.Lock()


def shared_calendar(**kwargs) -> NewsCalendar:
    """One calendar per configuration for the whole process, so several symbol
    bots do not each download the same feed."""
    key = tuple(sorted((k, v if not isinstance(v, list) else tuple(v))
                       for k, v in kwargs.items() if k not in ("fetch",)))
    with _shared_lock:
        cal = _shared.get(key)
        if cal is None:
            cal = _shared[key] = NewsCalendar(**kwargs)
        return cal
