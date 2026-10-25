"""
core/config/audit.py — who changed which parameter, when, and from what.

A live trading bot whose risk settings can be edited from a browser needs a
record that survives a restart. This writes to the same SQLite file the bot
already uses (``data/trading.db``) in its own table, so there is still exactly
one database file.

Every function here is fail-safe: an audit write that raises would take down a
settings change that has already been applied in memory, so failures are logged
and swallowed. The audit trail is important, but it must never be the reason the
bot stops trading.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS config_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    source     TEXT NOT NULL,
    key        TEXT NOT NULL,
    old_value  TEXT,
    new_value  TEXT
)
"""


def _connect(db_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute(_CREATE_TABLE)
    return conn


def _encode(value: Any) -> Optional[str]:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def record_changes(
    db_path: str,
    source: str,
    changes: Dict[str, Tuple[Any, Any]],
) -> None:
    """Append one row per changed key. Never raises."""
    if not changes:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        (now, source, key, _encode(old), _encode(new))
        for key, (old, new) in changes.items()
    ]
    try:
        with _connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO config_audit (ts, source, key, old_value, new_value) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("config audit write failed: %s", exc)


def recent_changes(db_path: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Newest-first audit entries. Returns ``[]`` if the DB is unreadable."""
    limit = max(1, min(int(limit), 1000))
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                "SELECT ts, source, key, old_value, new_value "
                "FROM config_audit ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            out = []
            for ts, source, key, old_value, new_value in cur.fetchall():
                out.append(
                    {
                        "ts": ts,
                        "source": source,
                        "key": key,
                        "old": _decode(old_value),
                        "new": _decode(new_value),
                    }
                )
            return out
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("config audit read failed: %s", exc)
        return []


def _decode(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
