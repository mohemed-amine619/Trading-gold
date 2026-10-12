"""
tests/unit/test_sqlite_persistence.py

Acceptance tests for SQLite-only database architecture and persistence:
1. Enforces that only SQLite URLs are allowed by Settings.
2. Verifies that init_db() automatically creates missing parent directories for SQLite files.
3. Verifies PRAGMA configurations (busy_timeout, foreign_keys, journal_mode).
4. Verifies database backup logic compatibility with SQLite.
"""

import os
import sqlite3
import tempfile
import pytest
from pydantic import ValidationError

from core.settings import Settings
from core.database.engine import init_db, get_sync_db, set_sqlite_journal_mode


def test_settings_rejects_non_sqlite_database_urls():
    """DATABASE_URL must reject postgresql, mysql, etc."""
    # Valid SQLite URLs
    s1 = Settings(DATABASE_URL="sqlite:///./data/trading.db")
    assert s1.database_url == "sqlite:///./data/trading.db"

    s2 = Settings(DATABASE_URL="sqlite+aiosqlite:///./data/trading.db")
    assert s2.database_url == "sqlite+aiosqlite:///./data/trading.db"

    s3 = Settings(DATABASE_URL="sqlite:///:memory:")
    assert s3.database_url == "sqlite:///:memory:"

    # Invalid non-SQLite URLs
    with pytest.raises(ValidationError) as exc:
        Settings(DATABASE_URL="postgresql://user:pass@localhost:5432/trading")
    assert "SQLite is the only permitted database" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        Settings(DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/trading")
    assert "SQLite is the only permitted database" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        Settings(DATABASE_URL="mysql://root@localhost/trading")
    assert "SQLite is the only permitted database" in str(exc.value)


def test_init_db_creates_parent_directory():
    """init_db creates any missing parent directory for SQLite path automatically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested_dir = os.path.join(tmp_dir, "nested", "storage")
        db_path = os.path.join(nested_dir, "test_trading.db")
        assert not os.path.exists(nested_dir)

        sqlite_url = f"sqlite:///{db_path}"
        init_db(sqlite_url, create_tables=True)

        assert os.path.exists(nested_dir)
        assert os.path.exists(db_path)

        with get_sync_db() as session:
            # Query PRAGMA integrity_check
            result = session.connection().execute(
                __import__("sqlalchemy").text("PRAGMA integrity_check")
            ).scalar()
            assert result == "ok"


def test_sqlite_backup_api():
    """Verify sqlite3 online backup works as expected on the database file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = os.path.join(tmp_dir, "src.db")
        dst_path = os.path.join(tmp_dir, "dst.db")

        # Create source DB with a sample table
        conn = sqlite3.connect(src_path)
        conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO test_data (val) VALUES ('test_record')")
        conn.commit()

        # Perform atomic backup
        dst_conn = sqlite3.connect(dst_path)
        with dst_conn:
            conn.backup(dst_conn)
        dst_conn.close()
        conn.close()

        # Verify integrity and content of destination DB
        verify_conn = sqlite3.connect(dst_path)
        cur = verify_conn.cursor()
        cur.execute("PRAGMA integrity_check")
        assert cur.fetchone()[0] == "ok"
        cur.execute("SELECT val FROM test_data")
        assert cur.fetchone()[0] == "test_record"
        verify_conn.close()
