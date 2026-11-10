"""
core/risk/daily_pnl.py — realised P&L of a calendar day, and the daily-target audit trail.

sqlite3 only: no SQLAlchemy, no MetaTrader5, no ``run.py``. Every public function
is fail-safe: it never raises into the trading loop. A failure is logged (at most
once a minute per cause, because the caller polls every tick) and reported as
"unknown" (``None`` / ``False``), never as zero.

What the ``trades`` table really contains (read from ``run.py``, not assumed)
-----------------------------------------------------------------------------
``run.py`` creates it lazily, on the first recorded close::

    CREATE TABLE IF NOT EXISTS trades
        (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT,
         pnl REAL, balance REAL)
    ALTER TABLE trades ADD COLUMN exit_reason TEXT      -- additive, may pre-exist
    INSERT INTO trades (timestamp, symbol, pnl, balance, exit_reason) VALUES (...)

* ``timestamp`` is ``datetime.now().isoformat()``: the LOCAL wall clock of the
  machine running the bot, naive (no UTC offset), ``YYYY-MM-DDTHH:MM:SS.ffffff``
  (the fraction is omitted when it is exactly zero). ``run.py`` rolls its own
  day over with ``date.today()``, which is the same local calendar. So the
  ``day`` passed here must be that local date, never a UTC date, or the two
  disagree for the hours around midnight when the host is not on UTC. Rows are
  matched with the string range ``[day, day + 1)``, which is exact for ISO text
  and for date-only values; a row whose timestamp is not ISO text is simply
  never matched.
* ``pnl`` is ``pos.profit + pos.swap`` taken from a position snapshot just before
  the close request (live), or the simulated floating P&L (dry-run).
  Commission and fee are not included and slippage after the snapshot is not
  reflected.
* There is NO ``magic`` column. ``symbol`` is the bot instance's configured
  symbol (``cfg["symbol"]``), not the position's.
* Only closes the bot itself sends are recorded: ``_record_trade_result`` is
  called from ``_close_position`` and from nowhere else. NOT recorded: positions
  closed by the broker on their stop-loss or take-profit, partial closes
  (``_partial_close``), and positions closed by hand in the terminal. Manual
  positions the bot itself closes (``_close_all`` closes every position of the
  symbol, and manual management can close them) ARE recorded, with no way to tell
  them from the bot's own trades.

Consequence: ``realized_pnl_for_day`` faithfully sums what is in the table, but the
table is an incomplete ledger. It can UNDER-report the day (stop-outs, take-profits
and partials are missing) and it can include manual closes. Do not treat it as the
authoritative realised P&L of a profit target; the broker's deal history is (see
``realized_from_deals``, which does the same sum over deal objects fetched by the
caller). The functions here are honest about their input; the gap is in what
``run.py`` writes.

``daily_target_events`` (the audit trail) is created lazily with
``CREATE TABLE IF NOT EXISTS`` in the same SQLite file, exactly like ``trades``.
It therefore lives outside Alembic, a known wart shared with ``trades``. It is
append-only: rows are inserted, never updated or deleted.

The reads open the file read-only (``mode=ro``) with a short busy timeout so a
writer can never stall the trading loop; the main database file is never created
or modified by a read. On a WAL-mode database (``core/database/engine.py`` enables
WAL) SQLite may still create empty ``-wal``/``-shm`` sidecar files for a read-only
connection; they are harmless and git-ignored.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from core.risk.daily_state import day_key, finite_or_none

logger = logging.getLogger(__name__)

__all__ = [
    "realized_pnl_for_day",
    "record_day_event",
    "latest_day_event",
    "realized_from_deals",
]

_TRADES_TABLE = "trades"
_EVENTS_TABLE = "daily_target_events"

_READ_TIMEOUT_S = 0.25    # reads sit in the trading loop: never wait long
_WRITE_TIMEOUT_S = 2.0    # audit writes are rare; still shorter than run.py's 10 s
_WARN_EVERY_S = 60.0

_EVENTS_DDL = (
    "CREATE TABLE IF NOT EXISTS daily_target_events ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "timestamp TEXT, day TEXT, state TEXT, "
    "realized REAL, floating REAL, net REAL, target REAL, reason TEXT)"
)
_EVENTS_COLUMNS = (
    "id", "timestamp", "day", "state", "realized", "floating", "net", "target", "reason",
)

_last_warned: Dict[str, float] = {}


def _warn(key: str, message: str, *args: Any) -> None:
    """WARNING at most once per ``_WARN_EVERY_S`` per ``key`` (callers poll)."""
    now = time.monotonic()
    if now - _last_warned.get(key, -1e9) >= _WARN_EVERY_S:
        _last_warned[key] = now
        logger.warning(message, *args)


def _as_date(day: Any) -> Optional[date]:
    """A calendar date from a ``date``, ``datetime`` or ISO ``YYYY-MM-DD`` text."""
    try:
        if isinstance(day, datetime):
            return day.date() if day == day else None  # NaT != NaT
        if isinstance(day, date):
            return day
        if isinstance(day, str):
            return date.fromisoformat(day.strip()[:10])
    except (ValueError, AttributeError):
        return None
    return None


def _connect_ro(path: str, timeout_s: float) -> sqlite3.Connection:
    """Read-only connection: opening never creates or modifies the file."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=timeout_s)


def _columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    """Column names of ``table`` (empty when the table does not exist)."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


# ─────────────────────────────────────────────────────────────────────────────
# Realised P&L from the trades table
# ─────────────────────────────────────────────────────────────────────────────

def realized_pnl_for_day(
    db_path: Any,
    day: Any,
    symbol: Optional[str] = None,
    magic: Optional[int] = None,
    *,
    timeout_s: float = _READ_TIMEOUT_S,
) -> Optional[float]:
    """Sum of ``trades.pnl`` for the local calendar ``day``, or None if unknown.

    * The file or the ``trades`` table does not exist yet: ``0.0``. No trade was
      recorded, which is genuinely zero.
    * The database exists but cannot be read (corrupt, locked past
      ``timeout_s``, unexpected schema, a ``pnl`` of that day that is NULL or not
      a number, an unusable ``day``): ``None``. Unknown is not zero.
    * ``symbol`` filters on the ``symbol`` column (``None`` = every symbol).
    * ``magic`` filters on a ``magic`` column. The table has none today (see the
      module docstring), so a requested ``magic`` on a table without that column
      cannot be honoured and the result is ``None``; pass ``magic=None`` to accept
      the unfiltered sum knowingly. Rows with a NULL ``magic`` match no magic.

    Read-only, short timeout, never raises. See the module docstring for the
    timezone of ``day`` and for why this ledger can under-report the day.
    """
    try:
        target_day = _as_date(day)
        wanted_magic = None if magic is None else finite_or_none(magic)
        if target_day is None or (magic is not None and wanted_magic is None):
            return None
        path = os.fspath(db_path)
        if not path:
            return None  # no database configured is not "no trades"
        if not os.path.exists(path):
            return 0.0
        if not os.path.isfile(path):
            return None
        start = target_day.isoformat()
        end = (target_day + timedelta(days=1)).isoformat()

        with closing(_connect_ro(path, timeout_s)) as conn:
            columns = _columns(conn, _TRADES_TABLE)
            if not columns:
                return 0.0
            needed = {"timestamp", "pnl"}
            if symbol is not None:
                needed.add("symbol")
            if magic is not None:
                needed.add("magic")
            missing = sorted(needed - columns)
            if missing:
                _warn(f"pnl-schema:{path}",
                      "realized_pnl_for_day: trades table lacks column(s) %s; "
                      "realised P&L is unknown", ", ".join(missing))
                return None
            sql = "SELECT pnl FROM trades WHERE timestamp >= ? AND timestamp < ?"
            params: list = [start, end]
            if symbol is not None:
                sql += " AND symbol = ?"
                params.append(symbol)
            if magic is not None:
                sql += " AND magic = ?"
                params.append(int(wanted_magic))
            rows = conn.execute(sql, params).fetchall()

        terms = []
        for (value,) in rows:
            number = finite_or_none(value)
            if number is None:
                _warn(f"pnl-row:{path}",
                      "realized_pnl_for_day: a trades row of %s has an unreadable pnl; "
                      "realised P&L is unknown", start)
                return None
            terms.append(number)
        return math.fsum(terms)
    except Exception as exc:  # noqa: BLE001 - fail-safe by contract
        _warn(f"pnl:{db_path}", "realized_pnl_for_day(%s) failed: %s", db_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Audit trail of daily-target transitions
# ─────────────────────────────────────────────────────────────────────────────

def record_day_event(
    db_path: Any,
    day: Any,
    state: Any,
    realized: Any,
    floating: Any,
    target: Any,
    reason: Any,
    ts: Any = None,
    *,
    timeout_s: float = _WRITE_TIMEOUT_S,
) -> bool:
    """Append one row to ``daily_target_events``. Returns True when it was written.

    Call it when ``DayDecision.changed`` is True. ``state`` may be a ``DayState``
    or its string. ``net`` is stored as ``realized + floating`` when both are
    known; an unknown number (None, NaN, inf) is stored as NULL, never as 0.
    ``ts`` is an optional ``datetime`` or text; the default is the local
    ``datetime.now().isoformat()``, the same clock as ``trades.timestamp``.

    The table is created on first use and rows are never updated or deleted.
    Never raises; a failure returns False (the caller may retry).
    """
    try:
        text_day = day_key(day)
        if text_day is None:
            return False
        r = finite_or_none(realized)
        f = finite_or_none(floating)
        t = finite_or_none(target)
        net = (r + f) if (r is not None and f is not None) else None
        if ts is None:
            stamp = datetime.now().isoformat()
        elif isinstance(ts, datetime):
            stamp = ts.isoformat()
        else:
            stamp = str(ts)
        state_text = str(getattr(state, "value", state))
        reason_text = None if reason is None else str(reason)

        path = os.fspath(db_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=timeout_s)) as conn:
            conn.execute(_EVENTS_DDL)
            conn.execute(
                "INSERT INTO daily_target_events "
                "(timestamp, day, state, realized, floating, net, target, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (stamp, text_day, state_text, r, f, net, t, reason_text),
            )
            conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001 - fail-safe by contract
        _warn(f"event-write:{db_path}", "record_day_event(%s) failed: %s", db_path, exc)
        return False


def latest_day_event(
    db_path: Any,
    day: Any,
    *,
    timeout_s: float = _READ_TIMEOUT_S,
) -> Optional[Dict[str, Any]]:
    """The most recent audit row for ``day`` as a dict, or None.

    None means no row, no file, no table, or an unreadable database: this is a
    best-effort restore aid, not a source of truth. The dict has the table's
    columns (``id, timestamp, day, state, realized, floating, net, target,
    reason``), so it is a valid argument for ``DailyTargetController.restore``.
    Read-only, short timeout, never raises.
    """
    try:
        text_day = day_key(day)
        if text_day is None:
            return None
        path = os.fspath(db_path)
        if not os.path.isfile(path):
            return None
        with closing(_connect_ro(path, timeout_s)) as conn:
            if not _columns(conn, _EVENTS_TABLE):
                return None
            cursor = conn.execute(
                f"SELECT {', '.join(_EVENTS_COLUMNS)} FROM daily_target_events "
                "WHERE day = ? ORDER BY id DESC LIMIT 1",
                (text_day,),
            )
            row = cursor.fetchone()
        return dict(zip(_EVENTS_COLUMNS, row)) if row is not None else None
    except Exception as exc:  # noqa: BLE001 - fail-safe by contract
        _warn(f"event-read:{db_path}", "latest_day_event(%s) failed: %s", db_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Realised P&L from broker deal history (the authoritative source)
# ─────────────────────────────────────────────────────────────────────────────

# MetaTrader5 DEAL_TYPE_BUY / DEAL_TYPE_SELL. Every other type (balance, credit,
# charge, correction, bonus, ...) is an account operation, not trading P&L: a
# deposit carries its amount in ``profit`` and must never be counted as a win.
_TRADE_DEAL_TYPES = (0.0, 1.0)


def realized_from_deals(
    deals: Optional[Iterable[Any]],
    magic: Optional[int] = None,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> Optional[float]:
    """Net P&L of the trade deals in ``deals``, or None when anything is unknown.

    ``deals`` are broker history-deal objects fetched by the caller (for example
    ``mt5.history_deals_get(...)``), duck-typed on ``type``, ``magic``, ``time``,
    ``profit``, ``commission``, ``swap`` and ``fee``. Unlike the ``trades`` table
    this sees broker-side stop-outs, take-profits and partial closes.

    * Only buy/sell deals count; balance operations are skipped. A deal whose
      ``type`` cannot be read makes the result None (it might be a deposit).
    * ``magic`` keeps deals with that ``.magic`` (an object with no ``.magic``
      counts as 0, as in ``run.py``); ``None`` keeps all.
    * ``start_ts`` / ``end_ts`` select ``start_ts <= deal.time < end_ts`` (either
      may be None). The caller chooses the day boundary and passes times in the
      same basis as ``deal.time`` (the broker server clock). A deal whose
      ``time`` is unreadable while a window is requested makes the result None.
    * ``profit`` must be finite; a missing or None ``commission``/``swap``/``fee``
      is 0, a present but non-finite one makes the result None.
    * Entry commissions are included: a cost paid today is part of today's result
      even when the position is still open.

    Pure (no I/O). ``deals`` being None (a failed broker call) is unknown, an
    empty sequence is a genuine 0.0. Never raises.

    Not verified against a live terminal (MetaTrader5 is Windows-only): the
    attribute names follow the documented ``TradeDeal`` fields. A closing deal
    sent from the terminal by hand may carry magic 0 even when the position was
    opened by the bot.
    """
    try:
        if deals is None:
            return None
        wanted = None if magic is None else finite_or_none(magic)
        if magic is not None and wanted is None:
            return None
        lo = None if start_ts is None else finite_or_none(start_ts)
        hi = None if end_ts is None else finite_or_none(end_ts)
        if (start_ts is not None and lo is None) or (end_ts is not None and hi is None):
            return None
        windowed = lo is not None or hi is not None

        terms = []
        for deal in list(deals):
            if wanted is not None:
                have = finite_or_none(getattr(deal, "magic", 0))
                if have is None or have != wanted:
                    continue
            if windowed:
                when = finite_or_none(getattr(deal, "time", None))
                if when is None:
                    return None
                if (lo is not None and when < lo) or (hi is not None and when >= hi):
                    continue
            kind = finite_or_none(getattr(deal, "type", None))
            if kind is None:
                return None
            if kind not in _TRADE_DEAL_TYPES:
                continue
            profit = finite_or_none(getattr(deal, "profit", None))
            if profit is None:
                return None
            terms.append(profit)
            for extra in ("commission", "swap", "fee"):
                raw = getattr(deal, extra, None)
                if raw is None:
                    continue
                value = finite_or_none(raw)
                if value is None:
                    return None
                terms.append(value)
        total = math.fsum(terms)
        return total if math.isfinite(total) else None
    except Exception as exc:  # noqa: BLE001 - fail-safe by contract
        _warn("deals", "realized_from_deals failed: %s", exc)
        return None
