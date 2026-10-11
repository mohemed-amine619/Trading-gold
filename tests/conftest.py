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
    Detach Settings from any real ``.env`` for the duration of each test.

    pydantic-settings resolves ``env_file`` at instantiation time, so pointing
    it at a path that cannot exist makes the model fall back to process
    environment variables and its own defaults only.
    """
    monkeypatch.setitem(
        Settings.model_config, "env_file", str(tmp_path / "__no_such_env__")
    )
    yield


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