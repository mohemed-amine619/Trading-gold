"""
tests/conftest.py — Shared pytest fixtures.

The critical job here is **hermetic settings**.

`core.settings.Settings` reads `.env` from the working directory. A developer
(or CI runner) who has a real `.env` will therefore get *different* Settings
values during tests than a clean checkout does. That silently makes tests
pass on one machine and fail on another — for example the live-trading
validation test reads DASHBOARD_SECRET_KEY from the developer's `.env` and no
longer sees the insecure default it is meant to assert on.

Fixing this properly means settings tests must not read the ambient `.env`.
"""

from __future__ import annotations

import pytest

from core.settings import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """
    Detach Settings and run.py from any real ``.env`` for the duration of each test.

    pydantic-settings resolves ``env_file`` at instantiation time, so pointing
    it at a path that cannot exist makes the model fall back to process
    environment variables and its own defaults only.
    """
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setitem(
        Settings.model_config, "env_file", str(tmp_path / "__no_such_env__")
    )
    yield


_REPO = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_runtime_paths(monkeypatch, tmp_path):
    """Keep every runtime file the bot writes out of the repository.

    On the Windows machine the repository IS the live bot's folder. run.py
    resolves the trades database (``DATABASE_URL``, default ``./data/trading.db``,
    which also locates ``journal.csv`` and ``scalp_report_*.json``) and the risk
    state (``RISK_STATE_DIR``, default ``data/state``) relative to the working
    directory, so running the suite there overwrote the live risk state - which
    can clear a drawdown halt - and added test trades to the table that feeds
    the daily target and the loss streak. Tests that need a specific path still
    set their own.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'trading.db'}")
    monkeypatch.setenv("RISK_STATE_DIR", str(tmp_path / "state"))
    yield


def _snapshot_live_files() -> dict:
    """Path -> (mtime, size) for the repo's live runtime files."""
    out = {}
    data = _REPO / "data"
    if data.exists():
        for p in data.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[str(p.relative_to(_REPO))] = (st.st_mtime_ns, st.st_size)
    env = _REPO / ".env"
    if env.exists():
        st = env.stat()
        out[".env"] = (st.st_mtime_ns, st.st_size)
    return out


@pytest.fixture(autouse=True, scope="session")
def live_files_untouched():
    """Fail the run if any test created or changed a file under the repo's
    ``data/`` or the repo's ``.env``: those are the live bot's files."""
    before = _snapshot_live_files()
    yield
    after = _snapshot_live_files()
    touched = sorted(k for k in after if before.get(k) != after[k])
    assert not touched, (
        "tests wrote to the live bot's files (use tmp_path / the isolated "
        f"runtime paths instead): {touched}")


@pytest.fixture
def clean_env(monkeypatch):
    """
    Remove every environment variable.

    Use instead of ``os.environ.clear()``: the latter also destroys PATH and
    HOME, which can break subprocess-based tooling for the rest of the session.
    """
    for key in list(__import__("os").environ):
        monkeypatch.delenv(key, raising=False)
    yield