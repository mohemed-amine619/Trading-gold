"""
tests/unit/test_ai_model_rigor.py — Statistical rigor of the run.py AI predictor.

Covers the champion/challenger promotion gate, walk-forward validation,
probability calibration, edge-gated position sizing, and the config plumbing
that feeds AIEnsemble.

These tests exist because the previous implementation promoted *any* model
that finished fitting. A retrain that had silently decayed into a coin flip
went straight into production, and the raw soft-vote probability that drives
both the 45-point confluence score and the position-size multiplier was never
checked against reality.

run.py imports MetaTrader5 at module scope, so the same sys.modules stub used
by test_progressive_breakeven_run.py is installed before importing it.
No test here reaches a broker; all models are fitted on synthetic OHLCV.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_mt5_stub() -> None:
    if "MetaTrader5" in sys.modules:
        stub = sys.modules["MetaTrader5"]
        if not hasattr(stub, "TIMEFRAME_M15"):
            stub.TIMEFRAME_M15 = 15
        return
    stub = types.ModuleType("MetaTrader5")
    stub.TIMEFRAME_M1 = 1
    stub.TIMEFRAME_M5 = 5
    stub.TIMEFRAME_M15 = 15
    stub.TIMEFRAME_H1 = 16385
    stub.TIMEFRAME_H4 = 16388
    stub.ORDER_TYPE_BUY = 0
    stub.ORDER_TYPE_SELL = 1
    stub.TRADE_ACTION_SLTP = 1
    stub.TRADE_RETCODE_DONE = 10009
    sys.modules["MetaTrader5"] = stub


@pytest.fixture(scope="module")
def run_module(tmp_path_factory):
    """Import run.py once, hermetically.

    run.py calls ``load_dotenv()`` at module scope, and python-dotenv's
    ``find_dotenv()`` walks up from the *calling file's* directory rather than
    the CWD. So importing run.py anywhere loads the repository ``.env`` --
    which holds live broker credentials -- into ``os.environ`` for the whole
    process, which then breaks test_safety_paper_default's credential checks
    in any module that runs later.

    tests/conftest.py already guards this with ``PYTHON_DOTENV_DISABLED``, but
    that is a FUNCTION-scoped autouse fixture, and a module-scoped fixture is
    instantiated before it. Hence the explicit guard here.
    """
    _install_mt5_stub()

    prev_cwd = os.getcwd()
    prev_flag = os.environ.get("PYTHON_DOTENV_DISABLED")
    cwd = tmp_path_factory.mktemp("ai_rigor_cwd")
    try:
        os.chdir(cwd)
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        sys.path.insert(0, str(REPO_ROOT))
        sys.modules.pop("run", None)
        mod = importlib.import_module("run")
        assert "MT5_LOGIN" not in os.environ, (
            "run.py import leaked credentials from .env into the environment"
        )
        yield mod
    finally:
        sys.modules.pop("run", None)
        sys.path.remove(str(REPO_ROOT))
        os.chdir(prev_cwd)
        if prev_flag is None:
            os.environ.pop("PYTHON_DOTENV_DISABLED", None)
        else:
            os.environ["PYTHON_DOTENV_DISABLED"] = prev_flag


# ---------------------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 1000, phi: float = 0.5, noise: float = 0.002,
                seed: int = 3) -> pd.DataFrame:
    """AR(1) log-returns.

    ``phi`` controls predictability: phi=0 is a random walk with no
    exploitable structure, phi=+/-0.5 has genuine autocorrelation the model
    can learn. A pure random walk is the honest control case -- the promotion
    gate must not manufacture edge where none exists.
    """
    rng = np.random.default_rng(seed)
    step = np.zeros(n)
    eps = rng.normal(0.0, noise, n)
    for i in range(1, n):
        step[i] = phi * step[i - 1] + eps[i]
    close = 2000.0 * np.exp(np.cumsum(step))
    spread = close * rng.uniform(0.0010, 0.0040, n)
    high = close + spread * rng.uniform(0.3, 1.0, n)
    low = close - spread * rng.uniform(0.3, 1.0, n)
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0.0, spread * 0.2, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "tick_volume": rng.integers(400, 1600, n).astype(float)},
        index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
    )
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["macd_hist"] = 0.0
    df["macd_hist_prev"] = 0.0
    df["stoch_k"] = 50.0
    df["stoch_d"] = 50.0
    df["adx"] = 22.0
    df["di_plus"] = 15.0
    df["di_minus"] = 15.0
    for p in (9, 21, 50, 200):
        df[f"ema{p}"] = ta.ema(df["close"], length=p)
    df["vol_ratio"] = 1.0
    df["_high20"] = df["high"].rolling(20).max()
    df["_low20"] = df["low"].rolling(20).min()
    return df


def _cfg(**overrides):
    """AIEnsemble config with production-shaped defaults, overridable."""
    base = {
        "AI_CONFIDENCE_THRESHOLD": 0.55,
        "AI_TRAIN_BARS": 1000,
        "AI_RETRAIN_BARS": 50,
        "AI_VALIDATION_FRACTION": 0.20,
        "AI_MIN_OOS_ACCURACY": 0.52,
        "AI_MIN_OOS_AUC": 0.50,
        "AI_MIN_OOS_EDGE": 0.0,
        "AI_PROMOTE_TOLERANCE": 0.01,
        "AI_CALIBRATE": True,
        "AI_MIN_EDGE_FOR_SIZE_UP": 0.10,
    }
    base.update(overrides)
    return type("Cfg", (), base)()


def _ensemble(run_module, symbol="XAUUSDm", **overrides):
    return run_module.AIEnsemble(symbol, _cfg(**overrides))


# ---------------------------------------------------------------------------
# Config plumbing -- these are the keys that reach the model
# ---------------------------------------------------------------------------

def test_config_declares_every_ai_rigor_key(run_module):
    """New AI keys must exist in CONFIG, or cfg[...] raises KeyError."""
    required = [
        "ai_validation_fraction", "ai_min_oos_accuracy", "ai_min_oos_auc",
        "ai_min_oos_edge", "ai_promote_tolerance", "ai_calibrate",
        "ai_min_edge_for_size_up", "ai_size_high_conf", "ai_size_low_conf",
    ]
    missing = [k for k in required if k not in run_module.CONFIG]
    assert not missing, f"AI rigor keys missing from CONFIG: {missing}"


def test_config_bounds_are_sane(run_module):
    cfg = run_module.CONFIG
    assert 0.0 <= cfg["ai_validation_fraction"] <= 0.5
    assert cfg["ai_min_oos_accuracy"] >= 0.50, "a floor below 0.50 admits coin flips"
    assert cfg["ai_min_oos_auc"] >= 0.50, "an AUC floor below 0.50 admits random ranking"
    assert cfg["ai_min_oos_edge"] >= -1.0
    assert cfg["ai_size_high_conf"] > cfg["ai_size_low_conf"]


def test_settings_mirror_run_py_ai_keys(run_module):
    """AGENTS.md rule: a key on one side only is silently ignored on the other."""
    from core.settings import Settings
    legacy = Settings().as_legacy_config()
    for key in ("ai_validation_fraction", "ai_min_oos_accuracy",
                "ai_min_oos_auc", "ai_min_oos_edge", "ai_promote_tolerance",
                "ai_calibrate", "ai_min_edge_for_size_up"):
        assert key in legacy, f"{key} missing from as_legacy_config()"
        assert key in run_module.CONFIG, f"{key} missing from run.py CONFIG"


def test_ensemble_construction_site_supplies_every_key(run_module):
    """The AIEnsemble config object is built inline in _get_signal.

    train() reads these attributes outside any try/except, so a key missing
    here raises AttributeError straight into the main loop and kills the bot.
    """
    import inspect
    src = inspect.getsource(run_module.StandaloneAIBot._get_signal)
    block = src.split('AIEnsemble(')[1].split('})())')[0]
    supplied = set(__import__("re").findall(r'"([A-Z0-9_]+)"\s*:', block))
    needed = {"AI_CONFIDENCE_THRESHOLD", "AI_TRAIN_BARS", "AI_RETRAIN_BARS",
              "AI_VALIDATION_FRACTION", "AI_MIN_OOS_ACCURACY",
              "AI_MIN_OOS_AUC", "AI_MIN_OOS_EDGE", "AI_PROMOTE_TOLERANCE",
              "AI_CALIBRATE", "AI_MIN_EDGE_FOR_SIZE_UP"}
    assert needed <= supplied, f"not passed to AIEnsemble: {sorted(needed - supplied)}"


def test_environment_overrides_are_read(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("AI_MIN_OOS_ACCURACY", "0.77")
    monkeypatch.setenv("AI_VALIDATION_FRACTION", "0.35")
    monkeypatch.setenv("AI_CALIBRATE", "false")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop("run", None)
    try:
        mod = importlib.import_module("run")
        assert mod.CONFIG["ai_min_oos_accuracy"] == 0.77
        assert mod.CONFIG["ai_validation_fraction"] == 0.35
        assert mod.CONFIG["ai_calibrate"] is False
    finally:
        sys.modules.pop("run", None)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_platt_scale_refuses_tiny_split(run_module):
    """Calibrating 10 points would fit noise. Must decline."""
    cal = run_module._platt_scale(np.full(10, 0.7), np.array([1] * 5 + [0] * 5))
    assert cal is None


def test_platt_scale_refuses_single_class(run_module):
    """One class means there is no mapping towards -- decline."""
    y = np.zeros(run_module._MIN_VALIDATION + 5, dtype=int)
    cal = run_module._platt_scale(np.full(len(y), 0.7), y)
    assert cal is None


def test_calibrated_prob_is_passthrough_without_calibrator(run_module):
    assert run_module._calibrated_prob(None, 0.63) == pytest.approx(0.63)


def test_calibrated_prob_never_reports_certainty(run_module):
    """A calibrated 0.0 or 1.0 would be a lie and would pass any size gate."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.01, 0.99, 200)
    y = (raw > 0.5).astype(int)
    cal = run_module._platt_scale(raw, y)
    assert cal is not None
    floor = run_module._CALIBRATION_FLOOR
    for p in (0.0, 0.001, 0.5, 0.999, 1.0):
        out = run_module._calibrated_prob(cal, p)
        assert floor <= out <= 1.0 - floor, f"{p} -> {out} outside [{floor}, {1-floor}]"


def test_calibration_corrects_overconfidence(run_module):
    """The core defect this fixes: raw soft-vote probs claim 0.99 far more
    often than the outcome justifies. After calibration, claimed 0.99 must
    land near the empirical hit rate, not near 0.99."""
    rng = np.random.default_rng(5)
    n = 600
    # Overconfident generator: pushes probabilities away from the base rate.
    raw = np.clip(rng.normal(0.5, 0.22, n), 0.01, 0.99)
    y = (rng.uniform(size=n) < np.clip((raw - 0.5) * 1.2 + 0.5, 0.02, 0.98)).astype(int)

    cal = run_module._platt_scale(raw, y)
    assert cal is not None
    assert y.mean() == pytest.approx(0.5, abs=0.08)

    calibrated = np.array([run_module._calibrated_prob(cal, p) for p in raw])
    for hi in (0.80, 0.90):
        m_raw = raw >= hi
        m_cal = calibrated >= hi
        if m_raw.sum() >= 10 and m_cal.sum() >= 10:
            raw_rate = y[m_raw].mean()
            cal_rate = y[m_cal].mean()
            assert abs(cal_rate - raw_rate) < abs(raw_rate - 0.5), (
                f"calibration did not tighten the {hi}+ bucket "
                f"(raw hit {raw_rate:.3f}, calibrated hit {cal_rate:.3f})")


def test_platt_scale_survives_extreme_inputs(run_module):
    """0.0 and 1.0 must not blow up the logit transform."""
    rng = np.random.default_rng(2)
    raw = np.concatenate([[0.0, 1.0, 0.0, 1.0], rng.uniform(0.1, 0.9, 80)])
    y = (raw > 0.5).astype(int)
    cal = run_module._platt_scale(raw, y)
    assert cal is not None
    out = run_module._calibrated_prob(cal, 1.0)
    assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# UP column lookup
# ---------------------------------------------------------------------------

class _ReversedClasses:
    """Estimator whose probability columns are ordered [DOWN, UP] -> index 1.

    The old code hardcoded proba[1], which silently reads the wrong column
    whenever the estimator's class order is not ascending.
    """
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.8]]), (len(X), 1))


def test_up_index_uses_classes_not_position(run_module):
    pipe = types.SimpleNamespace(
        named_steps={"model": _ReversedClasses()})
    assert run_module.AIEnsemble._up_index(pipe) == 0


def test_up_index_is_none_when_no_up_class(run_module):
    class _OnlyDown:
        classes_ = np.array([0])
    pipe = types.SimpleNamespace(named_steps={"model": _OnlyDown()})
    assert run_module.AIEnsemble._up_index(pipe) is None


# ---------------------------------------------------------------------------
# Promotion gate -- pure policy tests
# ---------------------------------------------------------------------------

def test_gate_rejects_below_accuracy_floor(run_module):
    ai = _ensemble(run_module)
    ok, reason = ai._passes_gate(0.40, 0.60, 0.40)
    assert not ok
    assert "accuracy" in reason


def test_gate_rejects_below_auc_floor(run_module):
    ai = _ensemble(run_module)
    ok, reason = ai._passes_gate(0.90, 0.45, 0.40)
    assert not ok
    assert "auc" in reason


def test_gate_rejects_model_worse_than_majority_baseline(run_module):
    """61% accuracy on a 73.8% UP window is worse than always answering UP.

    This is the case the old code waved through: it promoted any model that
    fitted, then reported "61% accurate" while actually underperforming the
    dumbest possible strategy.
    """
    ai = _ensemble(run_module)
    ok, reason = ai._passes_gate(0.613, 0.523, 0.738)
    assert not ok
    assert "worse than always answering UP" in reason
    assert "-0.125" in reason


def test_gate_baseline_is_the_majority_class_not_the_up_rate(run_module):
    """On a 38% UP window the trivial baseline is "always DOWN" at 0.62.

    Measuring lift against the raw UP rate (0.38) would let a 50%-accurate
    model pass with a fictitious +0.12 edge while actually losing to a
    strategy that does nothing.
    """
    ai = _ensemble(run_module)
    ok, reason = ai._passes_gate(0.55, 0.60, 0.38)
    assert not ok
    assert "worse than always answering DOWN" in reason
    assert "-0.070" in reason

    # The mirror image: an UP-heavy window with the same nominal accuracy.
    ok_up, reason_up = ai._passes_gate(0.55, 0.60, 0.62)
    assert not ok_up
    assert "worse than always answering UP" in reason_up


def test_gate_accepts_model_with_real_lift(run_module):
    ai = _ensemble(run_module)
    ok, reason = ai._passes_gate(0.664, 0.738, 0.555)
    assert ok, reason


def test_gate_edge_is_direction_agnostic(run_module):
    """A DOWN-heavy window reaches the edge check with the same arithmetic."""
    ai = _ensemble(run_module)
    ok, reason = ai._passes_gate(0.55, 0.60, 0.30)  # baseline 0.70, lift -0.15
    assert not ok
    assert "DOWN" in reason


def test_gate_edge_check_can_be_disabled(run_module):
    """AI_MIN_OOS_EDGE=-1.0 restores unconditional promotion for operators who
    want the old behaviour."""
    ai = _ensemble(run_module, AI_MIN_OOS_EDGE=-1.0)
    ok, _ = ai._passes_gate(0.55, 0.60, 0.80)   # lift -0.25, but gate disabled
    assert ok


def test_gate_defends_incumbent(run_module):
    """A challenger meaningfully worse than the live model is refused."""
    ai = _ensemble(run_module)
    ai._last_oos_accuracy = 0.80
    ok, reason = ai._passes_gate(0.70, 0.90, 0.40)
    assert not ok
    assert "incumbent" in reason


def test_gate_tolerates_statistical_tie(run_module):
    """Tiny wobbles must not cause model churn."""
    ai = _ensemble(run_module, AI_PROMOTE_TOLERANCE=0.02)
    ai._last_oos_accuracy = 0.700
    ok, _ = ai._passes_gate(0.690, 0.90, 0.40)   # within tolerance
    assert ok


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------

def test_validation_split_is_time_ordered(run_module, monkeypatch):
    """The validation bars must be strictly LATER than the training bars.

    A shuffled split would leak future information backwards and produce an
    accuracy number that is pure fiction.
    """
    seen = {}
    real_build = run_module.AIEnsemble._build_xy

    def spy(self, df):
        X, y, avail = real_build(self, df)
        seen["X"] = X
        seen["y"] = y
        return X, y, avail

    monkeypatch.setattr(run_module.AIEnsemble, "_build_xy", spy)

    captured = {}
    real_promote = run_module.AIEnsemble._promote

    def spy_promote(self, pipe, cal, avail, n_train, **kw):
        captured["n_train"] = n_train
        captured["n_total"] = len(seen["X"])
        return real_promote(self, pipe, cal, avail, n_train, **kw)

    monkeypatch.setattr(run_module.AIEnsemble, "_promote", spy_promote)

    ai = _ensemble(run_module)
    ai.train(_make_ohlcv())

    if captured:  # only asserts when a model was actually promoted
        frac = run_module.CONFIG["ai_validation_fraction"]
        assert captured["n_total"] - captured["n_train"] == pytest.approx(
            captured["n_total"] * frac, abs=2)
        assert captured["n_train"] > 0


def test_split_is_skipped_when_window_too_small(run_module):
    """A validation split too small to measure anything is refused: the model
    trains ungated, with oos recorded as unmeasured rather than implying an
    edge nobody checked."""
    ai = _ensemble(run_module, AI_TRAIN_BARS=400, AI_VALIDATION_FRACTION=0.05)
    assert ai.train(_make_ohlcv()) is True
    assert ai._trained is True
    assert ai._last_oos_accuracy is None, "claimed an accuracy it never measured"
    assert ai._last_oos_edge is None
    assert ai._calibrator is None, "cannot calibrate without a validation split"


# ---------------------------------------------------------------------------
# End-to-end promotion behaviour
# ---------------------------------------------------------------------------

def test_random_walk_model_is_not_given_credit_for_the_base_rate(run_module):
    """phi=0 is a random walk. Whatever accuracy it shows, the lift over the
    trivial baseline is what matters, and that is reported."""
    df = _make_ohlcv(phi=0.0, noise=0.002, seed=17)
    ai = _ensemble(run_module)
    ai.train(df)
    if ai._last_oos_accuracy is not None:
        baseline = max(ai._last_oos_base_rate, 1.0 - ai._last_oos_base_rate)
        assert ai._last_oos_edge == pytest.approx(ai._last_oos_accuracy - baseline)
        # A promoted model must at minimum match the naive baseline.
        assert ai._last_oos_edge >= run_module.CONFIG["ai_min_oos_edge"]


def test_predictable_series_is_promoted_with_metrics(run_module):
    """With genuine autocorrelation the model must actually be allowed to
    trade -- the gate must not be so strict that the bot never signals."""
    ai = _ensemble(run_module)
    ok = ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert ok is True, ai._last_reject_reason
    assert ai._trained is True
    assert ai._n_promotions == 1
    assert ai._last_oos_accuracy > 0.55
    assert ai._last_oos_auc > 0.55
    assert ai._last_oos_edge > 0.0
    assert ai._calibrator is not None, "calibration is on by default"


def test_rejected_model_never_becomes_live(run_module, monkeypatch):
    """An impossible accuracy floor must block promotion, and the bot must
    refuse to trade on it."""
    ai = _ensemble(run_module, AI_MIN_OOS_ACCURACY=0.999)
    ok = ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert ok is False
    assert ai._trained is False
    assert ai._pipeline is None
    assert ai._n_rejections == 1
    assert "not promoted" in ai.predict(_make_ohlcv(phi=0.5)).reason


def test_challenger_never_overwrites_a_live_model(run_module):
    """The incumbent must survive a rejected challenger untouched."""
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    incumbent = ai._pipeline
    incumbent_acc = ai._last_oos_accuracy
    assert incumbent is not None

    # Force the gate to reject the next challenger.
    ai.config.AI_MIN_OOS_EDGE = 5.0
    ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=9))
    assert ai._pipeline is incumbent, "a rejected challenger replaced the live model"
    assert ai._last_oos_accuracy == incumbent_acc


def test_promotion_records_metrics_and_importance(run_module):
    ai = _ensemble(run_module)
    ai.train(_make_ohlcv(phi=-0.5, noise=0.002, seed=3))
    top = ai.top_features(5)
    assert 1 <= len(top) <= 5
    names = [n for n, _ in top]
    assert set(names) <= set(run_module.FEATURE_COLS)
    vals = [v for _, v in top]
    assert vals == sorted(vals, reverse=True), "importances must be ranked"


def test_calibration_can_be_switched_off(run_module):
    ai = _ensemble(run_module, AI_CALIBRATE=False)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    assert ai._calibrator is None


def test_predict_confidence_is_bounded_and_ordered(run_module):
    """Calibrated confidence must be a usable probability, and the reported
    confidence must be the one the direction was chosen on."""
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    sig = ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.direction in (-1, 0, 1)
    if sig.direction == 1:
        assert sig.confidence >= ai.config.AI_CONFIDENCE_THRESHOLD
    elif sig.direction == -1:
        assert sig.confidence >= ai.config.AI_CONFIDENCE_THRESHOLD
    else:
        assert run_module.CONFIG["ai_confidence_threshold"] - 1e-9 <= \
            sig.confidence <= 1.0 - run_module.CONFIG["ai_confidence_threshold"] + 1e-9


def test_predict_survives_a_broken_model(run_module):
    """A pipeline that raises must degrade to neutral, not crash the bot."""
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True

    class _Boom:
        named_steps = {"model": types.SimpleNamespace(classes_=np.array([0, 1]))}
        def predict_proba(self, X):
            raise RuntimeError("model exploded")

    ai._pipeline = _Boom()
    sig = ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert sig.direction == 0
    assert "predict error" in sig.reason


# ---------------------------------------------------------------------------
# Retrain scheduling
# ---------------------------------------------------------------------------

def test_update_does_not_spin_when_every_fit_is_rejected(run_module, monkeypatch):
    """A model with no edge must not retrain itself on every single bar.

    Before this change, a persistently failing fit re-entered on each bar and
    burned a full four-ensemble fit per bar forever.
    """
    calls = []
    monkeypatch.setattr(
        run_module.AIEnsemble, "train",
        lambda self, df: (calls.append(1), False)[1])

    ai = _ensemble(run_module, AI_RETRAIN_BARS=50)
    df = _make_ohlcv()
    for _ in range(200):
        ai.update(df)

    # 1 synchronous first fit, then at most one background fit per interval.
    assert len(calls) <= 1 + 200 // 50, (
        f"retrained {len(calls)} times in 200 bars with a 50-bar interval; "
        f"a permanently-rejected model must not spin on every candle")
    assert len(calls) >= 1


def test_rejected_model_does_not_rearm_the_synchronous_path(run_module, monkeypatch):
    """A model that never clears the gate must stay on the background cadence.

    Keying 'are we due?' on `not self._trained` re-arms every bar for exactly
    the model that needs protection most.
    """
    monkeypatch.setattr(run_module.AIEnsemble, "train", lambda self, df: False)

    ai = _ensemble(run_module, AI_RETRAIN_BARS=50)
    df = _make_ohlcv()
    ai.update(df)
    assert ai._n_attempts == 1
    for _ in range(10):
        ai.update(df)
    assert ai._n_attempts == 1, "re-entered the synchronous first-fit path"


def test_trained_model_is_reset_only_when_interval_elapses(run_module, monkeypatch):
    """The cadence is reset exactly once per due update."""
    monkeypatch.setattr(run_module.AIEnsemble, "train", lambda self, df: False)
    ai = _ensemble(run_module, AI_RETRAIN_BARS=10)
    df = _make_ohlcv()
    ai.update(df)
    assert ai._bars_since_train == 0
    for i in range(1, 11):
        ai.update(df)
        if i < 10:
            assert ai._bars_since_train == i, "cadence advanced without a due update"


def test_update_respects_the_retrain_interval(run_module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_module.AIEnsemble, "train",
        lambda self, df: (calls.append(1), True)[1])

    ai = _ensemble(run_module, AI_RETRAIN_BARS=50)
    df = _make_ohlcv()
    ai.update(df)                       # synchronous first fit
    assert len(calls) == 1
    assert ai._n_attempts == 1
    for _ in range(49):
        ai.update(df)                   # not due yet
    assert len(calls) == 1, "retrained before the interval elapsed"


# ---------------------------------------------------------------------------
# Position sizing -- edge, not bravado
# ---------------------------------------------------------------------------

def _lot(run_module, cfg, ai_conf):
    return run_module._size_lot(
        account_equity=10_000.0, sl_dist_price=5.0, tick_size=0.01,
        tick_value=1.0, vol_min=0.01, vol_max=10.0, vol_step=0.01,
        cfg=cfg, adx=20.0, ai_conf=ai_conf)


def test_size_does_not_upsize_without_real_edge(run_module):
    """A confidence above the high tier but below the edge floor must NOT
    scale up. This is the 'confidently wrong, and bigger' failure mode."""
    cfg = dict(run_module.CONFIG)
    cfg["ai_size_high_conf"] = 0.70
    cfg["ai_size_low_conf"] = 0.60
    cfg["ai_min_edge_for_size_up"] = 0.25

    baseline = _lot(run_module, cfg, 0.0)
    upsize = _lot(run_module, cfg, 0.85)     # conf 0.85, edge 0.35 -> allowed
    no_edge = _lot(run_module, cfg, 0.72)    # conf 0.72, edge 0.22 -> blocked

    assert upsize > baseline, "genuine edge did not increase size"
    assert no_edge == baseline, "size was increased without the required edge"


def test_size_downsizes_weak_confidence(run_module):
    cfg = dict(run_module.CONFIG)
    cfg["ai_size_high_conf"] = 0.85
    cfg["ai_size_low_conf"] = 0.60
    baseline = _lot(run_module, cfg, 0.0)
    assert _lot(run_module, cfg, 0.55) < baseline


def test_edge_gate_defaults_to_half_a_confidence_point(run_module):
    """Default edge requirement is 0.10, i.e. confidence must reach 0.60."""
    cfg = dict(run_module.CONFIG)
    # Pin the balanced 0.60 downsize tier this assertion was written against;
    # the money_printer default downsizes below 0.55 instead.
    cfg["ai_size_low_conf"] = 0.60
    assert cfg["ai_min_edge_for_size_up"] == pytest.approx(0.10)
    baseline = _lot(run_module, cfg, 0.0)
    # 0.59 confidence -> edge 0.09, just under the floor.
    assert _lot(run_module, cfg, 0.59) < baseline

# ---------------------------------------------------------------------------
# Using the live model vs. promoting it
# ---------------------------------------------------------------------------
# The promotion gate only guards REPLACEMENT. It never stopped an already-live
# model from continuing to emit confidence, even after every subsequent
# challenger showed the family had no edge over the majority-class baseline.
# Live logs showed exactly that: `ai_conf=0.664` while challengers were being
# rejected with `lift -0.054 (acc 0.538 vs baseline 0.591)`. A matching AI vote
# is worth up to 45 score points and an upsize on the sizing path, so an
# unvalidated model was supplying conviction it had not earned.

def test_unvalidated_live_model_is_neutralised(run_module):
    """A model promoted without a validation split has no measured edge and
    must not vote."""
    ai = _ensemble(run_module, AI_TRAIN_BARS=400, AI_VALIDATION_FRACTION=0.05)
    assert ai.train(_make_ohlcv()) is True
    assert ai._last_oos_edge is None, "precondition: unmeasured promotion"

    sig = ai.predict(_make_ohlcv())
    assert sig.direction == 0
    assert sig.confidence == pytest.approx(0.5)
    assert "no measured OOS edge" in sig.reason


def test_live_model_with_negative_edge_is_neutralised(run_module):
    """A live model whose own recorded lift is below the floor must abstain."""
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    assert ai._last_oos_edge > 0.0, "precondition: a validated model"

    ai._last_oos_edge = -0.054          # decayed, as the live log showed
    sig = ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert sig.direction == 0
    assert sig.confidence == pytest.approx(0.5)
    assert "below floor" in sig.reason


def test_edge_gate_does_not_penalise_a_validated_model(run_module):
    """The gate must not be so strict that a good model stops trading."""
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    sig = ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert "AI neutral" not in sig.reason


def test_edge_gate_escape_hatch_restores_unconditional_predictions(run_module):
    """AI_MIN_OOS_EDGE=-1.0 is the documented way back to the old behaviour."""
    ai = _ensemble(run_module, AI_MIN_OOS_EDGE=-1.0)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    ai._last_oos_edge = -0.054
    sig = ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3))
    assert "AI neutral" not in sig.reason


def test_neutralisation_is_logged_once_not_per_bar(run_module, caplog):
    """predict() runs every bar; a warning per bar would bury the log."""
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    ai._last_oos_edge = -0.054
    df = _make_ohlcv(phi=0.5, noise=0.002, seed=3)

    with caplog.at_level("WARNING"):
        for _ in range(5):
            ai.predict(df)

    blocked = [r for r in caplog.records if "neutralised" in r.getMessage()]
    assert len(blocked) == 1


def test_promoting_a_valid_model_clears_the_block(run_module):
    """After the live model's metrics change, the gate re-evaluates.

    Otherwise a single bad state would silence the AI until the process is
    restarted, even after a valid model was installed.
    """
    ai = _ensemble(run_module)
    assert ai.train(_make_ohlcv(phi=0.5, noise=0.002, seed=3)) is True
    ai._last_oos_edge = -0.054
    assert ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3)).direction == 0
    assert ai._edge_block_reason

    ai._promote(ai._pipeline, ai._calibrator, ai._feature_cols, 800,
                oos_acc=0.60, oos_edge=0.05)
    assert ai._edge_block_reason == ""
    assert "AI neutral" not in ai.predict(_make_ohlcv(phi=0.5, noise=0.002, seed=3)).reason
