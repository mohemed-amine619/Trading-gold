"""
tests/unit/test_risk_ladder.py — capital preservation, Fibonacci scoring,
persisted risk state, and model persistence.

These cover the parts of the "never blow up" contract that are actually
enforceable:

* a restart must not re-arm today's loss budget,
* the ladder must pause and half-size *before* the hard cap,
* a hard halt must require an explicit reset,
* the confluence score must stay a 100-point budget no matter which
  components are enabled,
* a promoted model must survive a restart, and an incompatible file must be
  ignored rather than trusted.
"""

from __future__ import annotations

import os
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_bot(run_module, state_path, **overrides):
    cfg = dict(run_module.CONFIG)
    cfg.update({
        "symbol": "XAUUSDm",
        "dry_run": True,
        "risk_per_trade": 0.01,
        "max_daily_dd": 0.03,
        "max_weekly_dd": 0.10,
        "max_total_dd": 0.30,
        "halt_at_dd": 0.80,
        "reduce_size_at_dd": 0.50,
        "max_consecutive_losses": 3,
        "loss_streak_cooldown_minutes": 30,
        "equity_protect_close_all": False,
        "ai_model_persist": True,
    })
    cfg.update(overrides)
    bot = run_module.StandaloneAIBot(cfg)
    bot._risk_state_path = state_path
    return bot


def bull_frame():
    row = {
        "close": 2100.0, "ema9": 2101.0, "ema21": 2098.0, "ema50": 2095.0,
        "ema200": 2080.0, "rsi": 62.0, "macd_hist": 1.2, "macd_hist_prev": 0.9,
        "stoch_k": 65.0, "stoch_d": 58.0, "adx": 34.0, "atr": 8.0,
        "at_value_area": 0.8, "dist_ema21_atr": 0.4, "dist_ema50_atr": 0.8,
        "pinbar_wick": 0.1, "vol_ratio": 1.2,
    }
    return pd.DataFrame([row])


class FakeLocation:
    """The only surface ``_fib_score`` is allowed to depend on."""

    def __init__(self, position, valid=True, below=False, above=False):
        self.range_position = position
        self.is_valid = valid
        self.below_range = below
        self.above_range = above


# ─────────────────────────────────────────────────────────────────────────────
# Persisted risk state
# ─────────────────────────────────────────────────────────────────────────────

def test_restart_on_the_same_day_keeps_the_original_daily_start(
        run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    path = str(tmp_path / "risk.json")

    first = make_bot(run_module, path)
    first.daily_date = date.today()
    first.daily_start = 10_000.0
    first.trade_date = date.today()
    first.trades_today = 2
    first._persist_risk_state()

    # Simulate a restart: connect() would set daily_start from live equity.
    second = make_bot(run_module, path)
    second.daily_start = 9_000.0
    second._restore_risk_state(9_000.0)

    assert second.daily_start == 10_000.0, (
        "a restart must not reset the daily loss reference"
    )
    assert second.trades_today == 2


def test_restart_on_a_new_day_uses_live_equity(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    path = str(tmp_path / "risk.json")

    first = make_bot(run_module, path)
    first.daily_date = date(2020, 1, 1)   # stale on purpose
    first.daily_start = 10_000.0
    first._persist_risk_state()

    second = make_bot(run_module, path)
    second.daily_start = 9_500.0
    second._restore_risk_state(9_500.0)

    assert second.daily_start == 9_500.0


def test_peak_equity_is_never_reset_downward_by_a_restart(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    path = str(tmp_path / "risk.json")

    first = make_bot(run_module, path)
    first._peak_equity = 12_000.0
    first._persist_risk_state()

    second = make_bot(run_module, path)
    second._restore_risk_state(9_000.0)
    assert second._peak_equity == 12_000.0


def test_loss_streak_cooldown_survives_a_restart(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    path = str(tmp_path / "risk.json")

    first = make_bot(run_module, path)
    first._cooldown_until = run_module.datetime.utcnow() + run_module.timedelta(minutes=20)
    first._persist_risk_state()

    second = make_bot(run_module, path)
    second._restore_risk_state(10_000.0)
    assert second._cooldown_until is not None
    assert second._entry_gate(10_000.0)[0] is False


# ─────────────────────────────────────────────────────────────────────────────
# Capital-preservation ladder
# ─────────────────────────────────────────────────────────────────────────────

def test_ladder_halves_size_at_half_the_daily_budget(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"))
    bot.daily_date = date.today()
    bot.daily_start = 10_000.0
    bot._halt_reason = None

    # 1.5% drawdown on a 3% cap == 50% of the budget.
    allowed, mult, reason = bot._entry_gate(9_850.0)
    assert allowed is True
    assert mult == 0.5
    assert "half size" in reason


def test_ladder_pauses_entries_before_the_hard_cap(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"))
    bot.daily_date = date.today()
    bot.daily_start = 10_000.0

    # 2.4% of a 3% cap == 80% of the budget: no new risk, but not flattened.
    allowed, mult, reason = bot._entry_gate(9_760.0)
    assert allowed is False
    assert mult == 0.0
    assert "paused" in reason


def test_ladder_allows_full_size_below_the_first_rung(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"))
    bot.daily_date = date.today()
    bot.daily_start = 10_000.0

    allowed, mult, reason = bot._entry_gate(9_950.0)
    assert (allowed, mult, reason) == (True, 1.0, "")


def test_hard_daily_cap_returns_false(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"))
    bot.daily_date = date.today()
    bot.daily_start = 10_000.0

    # 4% drawdown on a 3% cap.
    assert bot._daily_dd_ok(9_600.0) is False


def test_weekly_drawdown_halts(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"),
                   max_weekly_dd=0.05, max_total_dd=0.90)
    bot._week_key = date.today().strftime("%G-W%V")
    bot._week_start_equity = 10_000.0

    # 6% below the week's start on a 5% weekly cap.
    assert bot._hard_drawdown_reason(9_400.0) is not None
    assert "weekly" in bot._halt_reason


def test_total_drawdown_halts_and_requires_a_manual_reset(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "r.json"),
                   max_weekly_dd=0.90, max_total_dd=0.20)
    bot._week_key = date.today().strftime("%G-W%V")
    bot._week_start_equity = 10_000.0
    bot._peak_equity = 10_000.0

    assert bot._hard_drawdown_reason(7_900.0) is not None
    assert "total DD" in bot._halt_reason

    # Recovering does not un-halt: the reset has to be deliberate.
    assert bot._entry_gate(9_900.0)[0] is False

    bot.reset_risk_halt("test")
    assert bot._halt_reason is None
    assert bot._entry_gate(9_900.0)[0] is True


def test_loss_streak_starts_a_cooldown(run_and_stub, tmp_path, monkeypatch):
    run_module, _ = run_and_stub
    monkeypatch.setattr(run_module, "_get_sqlite_db_path",
                        lambda: str(tmp_path / "trades.db"))
    bot = make_bot(run_module, str(tmp_path / "r.json"),
                   max_consecutive_losses=3, loss_streak_cooldown_minutes=30)

    for _ in range(2):
        bot._record_trade_result(-10.0)
    assert bot._cooldown_until is None

    bot._record_trade_result(-10.0)
    assert bot._cooldown_until is not None
    assert bot._consecutive_losses == 0, "counter resets after triggering"

    allowed, _mult, reason = bot._entry_gate(10_000.0)
    assert allowed is False
    assert "cooldown" in reason


def test_a_win_clears_the_loss_streak(run_and_stub, tmp_path, monkeypatch):
    run_module, _ = run_and_stub
    monkeypatch.setattr(run_module, "_get_sqlite_db_path",
                        lambda: str(tmp_path / "trades.db"))
    bot = make_bot(run_module, str(tmp_path / "r.json"), max_consecutive_losses=5)

    bot._record_trade_result(-10.0)
    bot._record_trade_result(-10.0)
    assert bot._consecutive_losses == 2

    bot._record_trade_result(+15.0)
    assert bot._consecutive_losses == 0


# ─────────────────────────────────────────────────────────────────────────────
# Fibonacci confluence
# ─────────────────────────────────────────────────────────────────────────────

def test_fib_score_peaks_at_the_golden_ratio(run_and_stub):
    run_module, _ = run_and_stub
    cfg = {"fib_zone_low": 0.382, "fib_zone_high": 0.786}

    # For a long, the 0.618 retracement of the up-leg is range position 0.382.
    at_golden = run_module._fib_score(FakeLocation(0.382), 1, cfg)
    at_edge = run_module._fib_score(FakeLocation(0.214), 1, cfg)
    at_mid = run_module._fib_score(FakeLocation(0.500), 1, cfg)

    assert at_golden > at_mid > at_edge
    assert at_golden == pytest.approx(1.0)


def test_fib_score_mirrors_for_shorts(run_and_stub):
    run_module, _ = run_and_stub
    cfg = {"fib_zone_low": 0.382, "fib_zone_high": 0.786}

    # For a short the golden pocket is the premium side of the same range.
    assert run_module._fib_score(FakeLocation(0.618), -1, cfg) == pytest.approx(1.0)
    assert run_module._fib_score(FakeLocation(0.618), 1, cfg) < 1.0


def test_fib_score_is_zero_without_a_range(run_and_stub):
    run_module, _ = run_and_stub
    cfg = {"fib_zone_low": 0.382, "fib_zone_high": 0.786}

    assert run_module._fib_score(None, 1, cfg) == 0.0
    assert run_module._fib_score(FakeLocation(0.5, valid=False), 1, cfg) == 0.0
    assert run_module._fib_score(FakeLocation(0.5, below=True), 1, cfg) == 0.0
    assert run_module._fib_score(FakeLocation(0.5, above=True), 1, cfg) == 0.0


def test_disabling_fib_reproduces_the_legacy_weights(run_and_stub):
    """Backward compatibility: with fib off the score must be the old formula."""
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg.update({
        "fib_enabled": False, "fib_weight": 10.0,
        "trend_weight": 30.0, "momentum_weight": 15.0,
        "adx_strong": 30.0, "min_risk_scale": 0.5, "max_risk_scale": 1.2,
        "adx_min": 20,
    })
    m15 = bull_frame()
    last = m15.iloc[-1]

    got = run_module._score_direction(1, 0.8, m15, m15, m15, 1, cfg, location=None)

    expected = (
        0.8 * 45.0
        + run_module._trend_score(m15, 1) * 10.0
        + run_module._trend_score(m15, 1) * 20.0
        + run_module._momentum_score(last, 1) * 15.0
        + run_module._entry_quality_score(last) * 10.0
    )
    # The ADX boost applies to the trend component only when ADX >= adx_strong.
    assert got == pytest.approx(expected, rel=1e-6)


def test_fib_adjusts_the_score_without_breaking_the_scale(run_and_stub):
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg.update({
        "fib_enabled": True, "fib_weight": 10.0,
        "trend_weight": 30.0, "momentum_weight": 15.0, "adx_strong": 30.0,
    })
    m15 = bull_frame()

    without_location = run_module._score_direction(
        1, 0.8, m15, m15, m15, 1, cfg, location=None)
    with_golden = run_module._score_direction(
        1, 0.8, m15, m15, m15, 1, cfg, location=FakeLocation(0.382))

    # The budget is renormalised, so enabling fib with no location slightly
    # reduces the score, and a golden-zone location must more than recover it.
    assert without_location < with_golden
    assert 0.0 <= without_location <= 100.0
    assert 0.0 <= with_golden <= 100.0


def test_score_never_exceeds_the_hundred_point_budget(run_and_stub):
    run_module, _ = run_and_stub
    cfg = dict(run_module.CONFIG)
    cfg.update({
        "fib_enabled": True, "fib_weight": 40.0,
        "trend_weight": 60.0, "momentum_weight": 40.0, "adx_strong": 10.0,
    })
    m15 = bull_frame()

    score = run_module._score_direction(
        1, 1.0, m15, m15, m15, 1, cfg, location=FakeLocation(0.382))
    assert 0.0 <= score <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────

def _model_config(persist=True, timeframe=15):
    return SimpleNamespace(
        AI_CONFIDENCE_THRESHOLD=0.55,
        AI_TRAIN_BARS=800,
        AI_RETRAIN_BARS=50,
        AI_VALIDATION_FRACTION=0.2,
        AI_MIN_OOS_ACCURACY=0.52,
        AI_MIN_OOS_AUC=0.5,
        AI_MIN_OOS_EDGE=0.0,
        AI_PROMOTE_TOLERANCE=0.01,
        AI_CALIBRATE=True,
        AI_MIN_EDGE_FOR_SIZE_UP=0.1,
        AI_MODEL_PERSIST=persist,
        MODEL_TIMEFRAME=timeframe,
    )


def test_saved_model_restores_into_a_fresh_ensemble(run_and_stub, tmp_path):
    run_module, _ = run_and_stub

    model = run_module.AIEnsemble("XAUUSDm", _model_config())
    model._trained = True
    model._pipeline = {"dummy": "pipeline"}
    model._feature_cols = run_module.FEATURE_COLS[:3]
    model._last_oos_accuracy = 0.58
    model._last_oos_auc = 0.61
    model._last_oos_edge = 0.04
    model._n_trains = 4
    model._n_promotions = 2

    assert model.save_to_disk(str(tmp_path), timeframe=15) is True

    fresh = run_module.AIEnsemble("XAUUSDm", _model_config())
    assert fresh.load_from_disk(str(tmp_path), timeframe=15) is True
    assert fresh._trained is True
    assert fresh._last_oos_accuracy == pytest.approx(0.58)
    assert fresh._last_oos_edge == pytest.approx(0.04)
    assert fresh._feature_cols == run_module.FEATURE_COLS[:3]
    assert fresh._n_trains == 4


def test_loading_a_missing_model_is_a_no_op(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    model = run_module.AIEnsemble("XAUUSDm", _model_config())
    assert model.load_from_disk(str(tmp_path), timeframe=15) is False
    assert model._trained is False


def test_an_incompatible_model_version_is_ignored(run_and_stub, tmp_path):
    import joblib
    run_module, _ = run_and_stub

    path = os.path.join(str(tmp_path), "XAUUSDm_15.joblib")
    joblib.dump({
        "version": 999,
        "symbol": "XAUUSDm",
        "feature_cols": run_module.FEATURE_COLS[:2],
        "pipeline": {"dummy": True},
    }, path)

    model = run_module.AIEnsemble("XAUUSDm", _model_config())
    assert model.load_from_disk(str(tmp_path), timeframe=15) is False
    assert model._trained is False


def test_a_model_needing_a_feature_we_no_longer_build_is_ignored(
        run_and_stub, tmp_path):
    import joblib
    run_module, _ = run_and_stub

    path = os.path.join(str(tmp_path), "XAUUSDm_15.joblib")
    joblib.dump({
        "version": run_module.AIEnsemble.MODEL_FORMAT_VERSION,
        "symbol": "XAUUSDm",
        "feature_cols": ["this_feature_was_removed"],
        "pipeline": {"dummy": True},
    }, path)

    model = run_module.AIEnsemble("XAUUSDm", _model_config())
    assert model.load_from_disk(str(tmp_path), timeframe=15) is False
    assert model._trained is False


def test_model_for_another_symbol_is_ignored(run_and_stub, tmp_path):
    import joblib
    run_module, _ = run_and_stub

    path = os.path.join(str(tmp_path), "XAUUSDm_15.joblib")
    joblib.dump({
        "version": run_module.AIEnsemble.MODEL_FORMAT_VERSION,
        "symbol": "EURUSD",
        "feature_cols": run_module.FEATURE_COLS[:2],
        "pipeline": {"dummy": True},
    }, path)

    model = run_module.AIEnsemble("XAUUSDm", _model_config())
    assert model.load_from_disk(str(tmp_path), timeframe=15) is False


def test_untrained_model_is_not_saved(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    model = run_module.AIEnsemble("XAUUSDm", _model_config())
    assert model.save_to_disk(str(tmp_path), timeframe=15) is False
    assert not os.path.exists(os.path.join(str(tmp_path), "XAUUSDm_15.joblib"))
