"""
tests/unit/test_safety_paper_default.py — Regression tests for trading-mode safety.

Background
----------
`run.py` gates every mutating operation (`_send_order`, `_close_position`,
`_close_all`, `_modify_sltp`, `_modify_sl`, `_partial_close`) on
``cfg["dry_run"]``.  That key was originally ABSENT from the ``CONFIG`` dict,
so ``cfg.get("dry_run")`` returned ``None`` — falsy — and the bot sent REAL
orders by default.

These tests lock in the corrected behaviour:
  * the default is PAPER (no real orders),
  * LIVE requires an explicit ``TRADING_MODE=live``,
  * an unrecognised mode fails SAFE to paper rather than open to live.

`run.py` does ``import MetaTrader5`` at module level, which is Windows-only,
so we inject a minimal stub into ``sys.modules`` to import it on Linux.
No real terminal is ever contacted.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPO_ROOT / "run.py"


def _install_mt5_stub() -> None:
    """Register a minimal MetaTrader5 stub so run.py imports on Linux."""
    if "MetaTrader5" in sys.modules:
        return
    stub = types.ModuleType("MetaTrader5")
    stub.TIMEFRAME_M15 = 15
    stub.TIMEFRAME_H1 = 16385
    stub.TIMEFRAME_H4 = 16388
    sys.modules["MetaTrader5"] = stub


@pytest.fixture
def load_run(tmp_path, monkeypatch):
    """Import run.py with a given TRADING_MODE, in an isolated CWD.

    run.py opens a FileHandler at import time, so each load runs in tmp_path
    to avoid writing trading_bot.log into the repository.
    """
    _install_mt5_stub()

    def _load(trading_mode: str | None = None):
        monkeypatch.chdir(tmp_path)
        if trading_mode is None:
            monkeypatch.delenv("TRADING_MODE", raising=False)
        else:
            monkeypatch.setenv("TRADING_MODE", trading_mode)
        monkeypatch.syspath_prepend(str(REPO_ROOT))
        sys.modules.pop("run", None)
        return importlib.import_module("run")

    yield _load
    sys.modules.pop("run", None)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_key_exists_in_config(load_run):
    """The gating key must actually be present — its absence was the bug."""
    run = load_run()
    assert "dry_run" in run.CONFIG


def test_defaults_to_paper_when_mode_unset(load_run):
    """No TRADING_MODE in the environment must NOT mean live trading."""
    run = load_run(trading_mode=None)
    assert run.CONFIG["dry_run"] is True, (
        "Bot must default to paper mode; unset TRADING_MODE must never send live orders"
    )


def test_explicit_paper_mode_is_paper(load_run):
    run = load_run("paper")
    assert run.CONFIG["dry_run"] is True


def test_backtest_mode_is_not_live(load_run):
    """backtest must not be treated as live trading."""
    run = load_run("backtest")
    assert run.CONFIG["dry_run"] is True


def test_explicit_live_mode_disables_dry_run(load_run):
    """Opting into live is the ONLY way to send real orders."""
    run = load_run("live")
    assert run.CONFIG["dry_run"] is False


@pytest.mark.parametrize("mode", ["LIVE", "Live", "  live  "])
def test_live_mode_is_case_and_whitespace_insensitive(load_run, mode):
    """A user typing 'Live' must not silently get paper mode."""
    run = load_run(mode)
    assert run.CONFIG["dry_run"] is False


@pytest.mark.parametrize("mode", ["", "yes", "true", "1", "live-ish", "liv"])
def test_unrecognised_mode_fails_safe_to_paper(load_run, mode):
    """Anything unrecognised must fall back to PAPER, never fail open to live."""
    run = load_run(mode)
    assert run.CONFIG["dry_run"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Static guarantees about the gate itself
# ─────────────────────────────────────────────────────────────────────────────

def _mutating_methods() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "StandaloneAIBot"
    )
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def _cfg_get_keys(fn: ast.FunctionDef) -> set[str]:
    """Keys read via cfg.get("...") inside a function (code only, not comments)."""
    keys = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def _self_calls(fn: ast.FunctionDef) -> set[str]:
    """Methods this function delegates to via self.<name>(...)."""
    out = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            out.add(node.func.attr)
    return out


def test_every_mutating_method_reaches_a_dry_run_gate():
    """
    Any method that can move money must reach a dry_run check — either
    directly, or by delegating only to gated mutators and read-only helpers.

    Checking *reachability* rather than a literal `dry_run` in the body is
    deliberate: `_close_all` calls the read-only `_open_positions` and then
    delegates to `_close_position`. Duplicating the guard there would be
    redundant. What must never happen is a path to a broker WRITE request with
    no gate anywhere along it.
    """
    gated_directly = {"_send_order", "_close_position", "_modify_sltp", "_modify_sl", "_partial_close"}
    delegating = {"_close_all"}
    # Read-only broker queries: legitimately called from inside a gated method.
    read_only_helpers = {"_open_positions", "_count_positions"}

    methods = _mutating_methods()

    for name in gated_directly:
        fn = methods.get(name)
        assert fn is not None, f"{name} not found in StandaloneAIBot"
        assert "dry_run" in _cfg_get_keys(fn), (
            f"{name} sends broker requests but never checks dry_run"
        )

    for name in delegating:
        fn = methods.get(name)
        assert fn is not None, f"{name} not found in StandaloneAIBot"
        for called in _self_calls(fn):
            if called in read_only_helpers:
                continue
            target = methods.get(called)
            assert target is not None, f"{name} calls unknown method {called}"
            assert "dry_run" in _cfg_get_keys(target), (
                f"{name} delegates to {called}, which is neither dry_run-gated "
                f"nor a known read-only helper — ungated path to a broker request"
            )


def test_dashboard_mode_reads_the_real_key():
    """
    The dashboard reported mode via cfg.get('paper_trading'), a key that
    exists nowhere in the codebase — so the badge always said LIVE.

    Checked against the parsed AST (not raw text) so that prose in comments
    cannot trip or mask the check.
    """
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
    read_keys = {
        node.args[0].value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
    }
    assert "paper_trading" not in read_keys, (
        "cfg.get('paper_trading') reads a key that exists nowhere in the codebase; "
        "mode reporting must use dry_run"
    )
    assert "dry_run" in read_keys


def test_no_live_default_in_trading_mode_resolution():
    """os.getenv default for TRADING_MODE must be 'paper', never 'live'."""
    source = RUN_PY.read_text(encoding="utf-8")
    assert 'os.getenv("TRADING_MODE", "paper")' in source
    assert 'os.getenv("TRADING_MODE", "live")' not in source


# ─────────────────────────────────────────────────────────────────────────────
# .env.example must document the switch
# ─────────────────────────────────────────────────────────────────────────────

def test_env_example_documents_trading_mode():
    env_example = REPO_ROOT / ".env.example"
    assert env_example.exists(), ".env.example is missing"
    content = env_example.read_text(encoding="utf-8")
    assert "TRADING_MODE" in content
    # The documented default must be the safe one.
    for line in content.splitlines():
        if line.strip().startswith("TRADING_MODE="):
            assert line.split("=", 1)[1].strip().lower() == "paper", (
                f".env.example must default TRADING_MODE to paper, got: {line!r}"
            )
