"""
core/database/engine.py — SQLAlchemy engine and session factory.

Supports both sync (for the trading loop) and async (for the API).
"""

from __future__ import annotations

import os
from contextlib import contextmanager, asynccontextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from core.database.models import Base

# SQLite journal mode override. ``None`` = platform default (WAL on POSIX,
# DELETE on Windows). Set ``enabled=True``/``False`` to force either mode.
_SQLITE_WAL: dict[str, bool] = {"enabled": None}


def set_sqlite_journal_mode(use_wal: bool) -> None:
    """
    Force SQLite WAL on or off.

    WAL improves concurrency but creates ``-wal``/``-shm`` sidecar files that
    Windows locks aggressively. The Windows default is therefore the DELETE
    journal; call this with ``False`` there if you see "database is locked".

    Must be called before :func:`init_db`.
    """
    _SQLITE_WAL["enabled"] = bool(use_wal)


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory
# ─────────────────────────────────────────────────────────────────────────────

_sync_engine = None
_async_engine = None
_SyncSession: sessionmaker | None = None
_AsyncSession: async_sessionmaker | None = None


def init_db(database_url: str, create_tables: bool = True) -> None:
    """
    Initialise the sync engine and optionally create all tables.
    Call once at application startup.

    Set ``create_tables=False`` when schema is owned by Alembic — the Docker
    entrypoint runs `alembic upgrade head` explicitly, and having the app
    silently create tables at import time would mask a missing migration.
    """
    global _sync_engine, _SyncSession

    # SQLite: enable WAL mode + foreign keys
    sync_url = database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    if "sqlite:///" in sync_url:
        db_path = sync_url.split("sqlite:///", 1)[1]
        if db_path and db_path != ":memory:":
            parent_dir = os.path.dirname(db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    _sync_engine = create_engine(
        sync_url,
        connect_args={"check_same_thread": False} if "sqlite" in sync_url else {},
        pool_pre_ping=True,
        echo=False,
    )

    if "sqlite" in sync_url:
        @event.listens_for(_sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            # WAL gives better concurrency but keeps -wal/-shm sidecar files
            # that Windows locks harder than POSIX. Use WAL only on POSIX, or
            # when the caller opts in for SQLite.
            #
            # On Windows the default DELETE journal avoids "database is locked"
            # failures when a second process (e.g. the GUI) opens the same file.
            use_wal = _SQLITE_WAL.get("enabled")
            if use_wal is None:
                use_wal = os.name != "nt"
            cursor.execute(f"PRAGMA journal_mode={'WAL' if use_wal else 'DELETE'}")
            cursor.execute("PRAGMA foreign_keys=ON")
            # NORMAL is safe with WAL; with the DELETE journal FULL is the
            # durable choice, since there is no WAL to recover from.
            cursor.execute(f"PRAGMA synchronous={'NORMAL' if use_wal else 'FULL'}")
            # Wait rather than immediately raising SQLITE_BUSY. Matters when a
            # backup or the GUI holds a write lock.
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    if create_tables:
        Base.metadata.create_all(_sync_engine)
    _SyncSession = sessionmaker(bind=_sync_engine, autoflush=True, autocommit=False)


async def init_async_db(database_url: str, create_tables: bool = True) -> None:
    """
    Initialise the async engine (used by the FastAPI layer).
    Converts a sync SQLite URL to async automatically.

    Set ``create_tables=False`` when schema is owned by Alembic.
    """
    global _async_engine, _AsyncSession

    async_url = database_url
    if "sqlite:///" in async_url and "+aiosqlite" not in async_url:
        async_url = async_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    if "sqlite" in async_url and ":///" in async_url:
        db_path = async_url.split(":///", 1)[1]
        if db_path and db_path != ":memory:":
            parent_dir = os.path.dirname(db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    _async_engine = create_async_engine(
        async_url,
        connect_args={"check_same_thread": False} if "sqlite" in async_url else {},
        pool_pre_ping=True,
        echo=False,
    )
    _AsyncSession = async_sessionmaker(
        bind=_async_engine, class_=AsyncSession, expire_on_commit=False
    )
    if create_tables:
        async with _async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


# ─────────────────────────────────────────────────────────────────────────────
# Session context managers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """Sync session context manager for use in the trading loop."""
    if _SyncSession is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    session: Session = _SyncSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_db():
    """Async session context manager for FastAPI dependency injection."""
    if _AsyncSession is None:
        raise RuntimeError("Async database not initialised. Call init_async_db() first.")
    async with _AsyncSession() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_dependency():
    """FastAPI dependency that yields an async session."""
    async with get_async_db() as session:
        yield session
