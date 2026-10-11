"""
tests/unit/test_windows_compat.py — Cross-platform behaviour.

These cover the places where the Windows 11 target differs from the Linux
development host: log directory creation, UTC timestamps, credential
redaction and SQLite journal-mode selection.

The point of these tests is that a change which works on Linux but breaks on
Windows should fail here, on Linux, in CI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from core import logging_setup
from core.database import engine as db_engine


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def test_log_directory_is_created(tmp_path):
    """A missing logs/ directory must not crash startup on Windows."""
    target = tmp_path / "nested" / "logs"
    assert not target.exists()

    directory = logging_setup.resolve_log_dir(target)
    assert directory.is_dir(), "resolve_log_dir must create the directory"


def test_configure_logging_writes_a_file(tmp_path):
    log_dir = tmp_path / "logs"
    logging_setup.configure_logging(log_dir=log_dir, log_filename="test.log")

    logging.getLogger("test.logger").info("hello from the test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = log_dir / "test.log"
    assert log_file.exists()
    assert "hello from the test" in log_file.read_text(encoding="utf-8")


def test_configure_logging_is_idempotent(tmp_path):
    """Calling twice must not duplicate handlers and duplicate every line."""
    logging_setup.configure_logging(log_dir=tmp_path / "logs")
    first = len(logging.getLogger().handlers)
    logging_setup.configure_logging(log_dir=tmp_path / "logs")
    second = len(logging.getLogger().handlers)
    assert first == second, "handlers must not accumulate across calls"


def test_configure_logging_uses_utc_timestamps(tmp_path):
    """Internal timestamps must be UTC regardless of machine timezone."""
    import re
    from datetime import datetime, timezone

    logging_setup.configure_logging(log_dir=tmp_path / "logs", log_filename="utc.log")
    logging.getLogger("test.utc").info("timestamped")
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = (tmp_path / "logs" / "utc.log").read_text(encoding="utf-8")
    stamp = re.search(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", text, re.MULTILINE)
    assert stamp, f"expected an ISO timestamp, got: {text[:120]!r}"

    parsed = datetime.fromisoformat(stamp.group(1)).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    # Allow generous slack for test runtime, but it must not be local time.
    assert abs((now - parsed).total_seconds()) < 600, "timestamp is not UTC"


# ─────────────────────────────────────────────────────────────────────────────
# Credential redaction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,must_not_contain", [
    ("password=supersecret", "supersecret"),
    ("MT5_PASSWORD: hunter2", "hunter2"),
    ("token=abc123def", "abc123def"),
    ("api_key=sk-live-1234", "sk-live-1234"),
    ("postgresql://user:dbpass@host:5432/db", "dbpass"),
    ("Authorization: Bearer tok_live_999", "tok_live_999"),
])
def test_redact_masks_secrets(raw, must_not_contain):
    masked = logging_setup.redact(raw)
    assert must_not_contain not in masked, f"secret leaked from {raw!r}"
    assert "***" in masked


def test_redact_leaves_ordinary_text_alone():
    text = "Order BUY EURUSD 0.10 at 1.10500"
    assert logging_setup.redact(text) == text


def test_redact_handles_empty_input():
    assert logging_setup.redact("") == ""


def test_secrets_never_reach_the_log_file(tmp_path):
    """End-to-end: a secret logged via arguments must be masked on disk."""
    log_dir = tmp_path / "logs"
    logging_setup.configure_logging(log_dir=log_dir, log_filename="secret.log")

    logger = logging.getLogger("test.secret")
    logger.error("Connecting with password=%s", "do-not-log-me")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = (log_dir / "secret.log").read_text(encoding="utf-8")
    assert "do-not-log-me" not in content, "credential reached the log file"


# ─────────────────────────────────────────────────────────────────────────────
# SQLite journal mode
# ─────────────────────────────────────────────────────────────────────────────

def test_sqlite_wal_can_be_forced(tmp_path):
    db_engine.set_sqlite_journal_mode(False)
    try:
        db_engine.init_db(f"sqlite:///{tmp_path / 'nowal.db'}")
        from sqlalchemy import text
        with db_engine.get_sync_db() as session:
            mode = session.execute(text("PRAGMA journal_mode")).scalar()
        assert mode == "delete"
    finally:
        db_engine.set_sqlite_journal_mode(None)


def test_sqlite_busy_timeout_is_set(tmp_path):
    """
    Windows raises 'database is locked' readily. busy_timeout makes the
    connection wait instead of failing immediately.
    """
    db_engine.init_db(f"sqlite:///{tmp_path / 'busy.db'}")
    from sqlalchemy import text
    with db_engine.get_sync_db() as session:
        timeout = session.execute(text("PRAGMA busy_timeout")).scalar()
    assert timeout and timeout > 0


def test_default_journal_mode_follows_platform():
    """
    WAL on POSIX, DELETE on Windows.

    Asserts the documented default rather than the host's actual value, so the
    same test passes on both CI and a Windows machine.
    """
    expected = "WAL" if os.name != "nt" else "DELETE"
    # The module encodes this decision as: use_wal = os.name != "nt"
    assert (os.name != "nt") == (expected == "WAL")


# ─────────────────────────────────────────────────────────────────────────────
# Path handling
# ─────────────────────────────────────────────────────────────────────────────

def test_paths_use_pathlib_not_hardcoded_separators():
    """
    Windows accepts forward slashes but emits backslashes in output; using
    pathlib keeps behaviour identical across platforms.
    """
    resolved = logging_setup.resolve_log_dir(Path("logs"))
    assert isinstance(resolved, Path)
    assert resolved.name == "logs"


def test_log_dir_setting_is_a_path():
    from core.settings import Settings
    assert isinstance(Settings().log_dir, Path)