"""
tests/unit/test_phase1_wiring.py — run.py wiring for the phase-1 changes.

The pure logic is tested in test_ai_intelligence_phase1.py. These tests check
that run.py actually uses it:

* the barrier label is selected only when asked for, and the close label is
  otherwise byte-for-byte the legacy path;
* the purge gap follows the label horizon;
* a model trained on other labels is not restored from disk;
* closed trades record their exit reason, a restart reloads the recent P&L
  history, and the streak brake is off unless enabled.

Database writes are redirected to tmp_path. Nothing here touches data/ in the
repository or a broker.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from tests.unit.test_ai_model_rigor import _ensemble, _make_ohlcv, run_module  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Label selection
# ─────────────────────────────────────────────────────────────────────────────

def test_label_mode_defaults_to_the_legacy_close_label(run_module):
    assert run_module.CONFIG["ai_label_mode"] == "close"
    assert _ensemble(run_module)._label_mode() == "close"


def test_unknown_label_mode_falls_back_to_legacy(run_module):
    ai = _ensemble(run_module, AI_LABEL_MODE="moonshot")
    assert ai._label_mode() == "close"


def test_barrier_mode_is_used_only_when_requested(run_module, monkeypatch):
    calls = []
    real = run_module._first_barrier_labels

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "_first_barrier_labels", spy)

    _ensemble(run_module)._build_xy(_make_ohlcv())
    assert calls == [], "close mode must not touch the barrier labeller"

    ai = _ensemble(run_module, AI_LABEL_MODE="barrier", AI_LABEL_HORIZON=7,
                   AI_SL_ATR_MULT=1.3)
    X, y, _ = ai._build_xy(_make_ohlcv())
    assert len(calls) == 1
    _, _, _, _, barrier_mult, horizon = calls[0]
    assert barrier_mult == pytest.approx(1.3), "barrier must be the real stop distance"
    assert horizon == 7
    assert X is not None and set(pd.unique(y)) <= {0, 1}


def test_barrier_mode_falls_back_when_helper_is_missing(run_module, monkeypatch):
    monkeypatch.setattr(run_module, "_first_barrier_labels", None)
    ai = _ensemble(run_module, AI_LABEL_MODE="barrier")
    assert ai._label_mode() == "close"
    assert ai._label_horizon() == 5


# ─────────────────────────────────────────────────────────────────────────────
# Purge gap
# ─────────────────────────────────────────────────────────────────────────────

def test_purge_gap_is_the_close_label_horizon_by_default(run_module):
    assert _ensemble(run_module)._label_horizon() == 5


def test_purge_gap_follows_the_barrier_horizon(run_module):
    ai = _ensemble(run_module, AI_LABEL_MODE="barrier", AI_LABEL_HORIZON=9)
    assert ai._label_horizon() == 9


def test_label_signature_separates_label_definitions(run_module):
    close = _ensemble(run_module)._label_signature()
    barrier = _ensemble(run_module, AI_LABEL_MODE="barrier",
                        AI_LABEL_HORIZON=12)._label_signature()
    assert close == "close:5"
    assert barrier == "barrier:12"


# ─────────────────────────────────────────────────────────────────────────────
# Persisted model guard
# ─────────────────────────────────────────────────────────────────────────────

def test_model_trained_on_other_labels_is_not_restored(run_module, tmp_path):
    trainer = _ensemble(run_module, AI_MODEL_PERSIST=True)
    assert trainer.train(_make_ohlcv()) is True
    assert trainer.save_to_disk(str(tmp_path), "M5") is True

    same_labels = _ensemble(run_module)
    assert same_labels.load_from_disk(str(tmp_path), "M5") is True

    other_labels = _ensemble(run_module, AI_LABEL_MODE="barrier", AI_LABEL_HORIZON=12)
    assert other_labels.load_from_disk(str(tmp_path), "M5") is False
    assert other_labels._trained is False


# ─────────────────────────────────────────────────────────────────────────────
# Trade record, restart seeding, streak brake
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bot(run_module, tmp_path, monkeypatch):
    """A StandaloneAIBot whose database lives in tmp_path."""
    db = str(tmp_path / "trading.db")
    monkeypatch.setattr(run_module, "_get_sqlite_db_path", lambda: db)
    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "XAUUSDm"
    cfg["dry_run"] = True
    return run_module.StandaloneAIBot(cfg)


def _rows(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT symbol, pnl, exit_reason FROM trades ORDER BY id").fetchall()
    finally:
        conn.close()


def test_closed_trade_records_its_exit_reason(bot, run_module):
    bot._record_trade_result(-1.5, "hard stop-loss")
    bot._record_trade_result(2.0, "profit lock giveback")
    db = run_module._get_sqlite_db_path()
    assert _rows(db) == [
        ("XAUUSDm", -1.5, "hard stop-loss"),
        ("XAUUSDm", 2.0, "profit lock giveback"),
    ]
    assert bot._recent_pnls == [-1.5, 2.0]


def test_existing_database_without_exit_reason_is_migrated(bot, run_module, tmp_path):
    db = run_module._get_sqlite_db_path()
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "timestamp TEXT, symbol TEXT, pnl REAL, balance REAL)")
    conn.execute("INSERT INTO trades (timestamp, symbol, pnl, balance) "
                 "VALUES ('2026-01-01', 'XAUUSDm', -3.0, 100.0)")
    conn.commit()
    conn.close()

    bot._record_trade_result(1.0, "age timeout")
    assert _rows(db) == [("XAUUSDm", -3.0, None), ("XAUUSDm", 1.0, "age timeout")]


def test_restart_reloads_recent_pnl_for_this_symbol_only(bot, run_module, monkeypatch):
    db = run_module._get_sqlite_db_path()
    bot._record_trade_result(-1.0, "stop")
    bot._record_trade_result(-1.0, "stop")
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO trades (timestamp, symbol, pnl, balance, exit_reason) "
                 "VALUES ('t', 'EURUSDm', 9.0, 0, 'x')")
    conn.commit()
    conn.close()

    cfg = dict(run_module.CONFIG)
    cfg["symbol"] = "XAUUSDm"
    cfg["dry_run"] = True
    restarted = run_module.StandaloneAIBot(cfg)
    assert restarted._recent_pnls == [-1.0, -1.0]


def test_streak_brake_is_off_by_default(bot):
    bot._recent_pnls = [-1.0, -1.0]
    bot.cfg["streak_sizing"] = False
    assert bot._streak_size_factor() == 1.0


def test_streak_brake_halves_size_after_two_losses_when_enabled(bot):
    bot._recent_pnls = [-1.0, -1.0]
    bot.cfg["streak_sizing"] = True
    assert bot._streak_size_factor() == 0.5


def test_streak_brake_clears_after_a_win_when_enabled(bot):
    bot._recent_pnls = [-1.0, -1.0, 3.0]
    bot.cfg["streak_sizing"] = True
    assert bot._streak_size_factor() == 1.0


def test_recent_history_is_bounded(bot, run_module):
    for _ in range(run_module._STREAK_HISTORY + 5):
        bot._record_trade_result(-1.0, "stop")
    assert len(bot._recent_pnls) == run_module._STREAK_HISTORY


def test_streak_sizing_flag_is_declared_for_the_dashboard(run_module):
    from core.config.schema import PARAMS
    assert "streak_sizing" in PARAMS
    assert "ai_label_mode" in PARAMS
    assert "ai_label_horizon" in PARAMS
    assert "streak_sizing" in run_module.CONFIG
