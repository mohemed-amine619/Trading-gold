"""
tests/unit/test_strategy_analysis_wiring.py — run.py wiring of the strategy panel and
AI strategy features. The panel must never change the score; the AI's features are
unchanged unless AI_STRATEGY_FEATURES is on.
"""

from __future__ import annotations

import types

import pytest

from core.strategy.features import COMPACT_COLUMNS
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import bull_frame
from tests.unit.test_strategy_library import market


def _ai(run_module, on):
    cfg = types.SimpleNamespace(AI_STRATEGY_FEATURES=on)
    return run_module.AIEnsemble("XAUUSDm", cfg)


def test_config_keys_default_off(run_and_stub):
    run_module, _ = run_and_stub
    assert run_module.CONFIG["strategy_panel_enabled"] is True
    assert run_module.CONFIG["ai_strategy_features"] is False
    assert run_module.CONFIG["nn_strategy_features"] is False


def test_ai_feature_parity_when_off(run_and_stub):
    run_module, _ = run_and_stub
    ai = _ai(run_module, False)
    assert ai._all_feature_cols() == run_module.FEATURE_COLS
    df = market(600, seed=4)
    assert ai._features(df).equals(run_module.build_features(df))


def test_ai_features_extended_when_on(run_and_stub):
    run_module, _ = run_and_stub
    ai = _ai(run_module, True)
    assert ai._all_feature_cols() == run_module.FEATURE_COLS + list(COMPACT_COLUMNS)
    feat = ai._features(market(600, seed=4))
    for c in COMPACT_COLUMNS:
        assert c in feat.columns


def test_score_has_no_panel_input(run_and_stub):
    """_score_direction takes no panel argument, and a built panel leaves it unchanged."""
    import inspect
    run_module, _ = run_and_stub
    params = inspect.signature(run_module._score_direction).parameters
    assert not any("panel" in p or "strateg" in p for p in params)
    m = bull_frame()
    cfg = dict(run_module.CONFIG)
    before = run_module._score_direction(1, 0.8, m, m, m, 1, cfg, location=None)
    from core.strategy.panel import build_panel
    build_panel(market(800, seed=9), "M15")
    after = run_module._score_direction(1, 0.8, m, m, m, 1, cfg, location=None)
    assert before == after


def test_panel_builder_runs_in_background_and_stores(run_and_stub, monkeypatch, tmp_path):
    run_module, _ = run_and_stub
    monkeypatch.setattr(run_module, "_get_sqlite_db_path", lambda: str(tmp_path / "x.db"))
    bot = types.SimpleNamespace(cfg=dict(run_module.CONFIG, symbol="XAUUSDm"),
                                _panel_builder=None, _panel_bar=None, _panel_thread=None,
                                _strategy_panels={}, _scalp_events=lambda: ())
    cls = getattr(run_module, "StandaloneAIBot", None)
    if cls is None:
        pytest.skip("bot class name differs")
    bot._build_strategy_panels = lambda frames, path=None: cls._build_strategy_panels(bot, frames, path)
    df = market(700, seed=6)
    cls._update_strategy_panel(bot, df, df, None)
    bot._panel_thread.join(60)
    assert set(bot._strategy_panels) == {"M15", "H1"}
    assert (tmp_path / "strategy_panel_XAUUSDm.json").exists()
    t = bot._panel_thread
    cls._update_strategy_panel(bot, df, df, None)        # same bar: no rebuild
    assert bot._panel_thread is t


def test_panel_disabled_does_nothing(run_and_stub):
    run_module, _ = run_and_stub
    cls = getattr(run_module, "StandaloneAIBot", None)
    if cls is None:
        pytest.skip("bot class name differs")
    bot = types.SimpleNamespace(cfg={"strategy_panel_enabled": False}, _panel_bar=None,
                                _panel_thread=None)
    cls._update_strategy_panel(bot, market(300), None, None)
    assert bot._panel_thread is None
