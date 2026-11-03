"""
core/risk/news_blackout.py — refuse NEW entries around scheduled high-impact news.

The user lists event times (UTC) in ``NEWS_BLACKOUT``; for ``before`` minutes ahead of an
event and ``after`` minutes behind it no new position is opened. Open positions are still
managed — the blackout only stops entries, it never closes or loosens anything.

Spread and slippage on gold balloon around releases such as NFP or CPI, and a 1 ATR stop is
not a 1 ATR loss there. The bot does not know the calendar, so the user supplies it.

Format: comma or semicolon separated ``YYYY-MM-DD HH:MM`` (UTC). A malformed item is
REPORTED, never silently dropped: a typo that disables a blackout the user thinks is on is
worse than a loud error. Pure functions, no broker, no clock of their own.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple


def parse_events(text: Optional[str]) -> Tuple[List[datetime], List[str]]:
    """Returns (events sorted, bad items). Empty/None text means no blackout."""
    events: List[datetime] = []
    bad: List[str] = []
    for raw in filter(None, (p.strip() for p in re.split(r"[;,]", text or ""))):
        item = raw.replace("T", " ")
        try:
            events.append(datetime.strptime(item, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
        except ValueError:
            bad.append(raw)
    return sorted(events), bad


def active_event(now: datetime, events: List[datetime], before_min: float,
                 after_min: float) -> Optional[datetime]:
    """The event whose window contains ``now`` (window edges inclusive), else None.
    ``now`` must be timezone-aware; a naive time is refused rather than assumed UTC."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (UTC)")
    before, after = timedelta(minutes=max(0.0, before_min)), timedelta(minutes=max(0.0, after_min))
    for ev in events:
        if ev - before <= now <= ev + after:
            return ev
    return None
