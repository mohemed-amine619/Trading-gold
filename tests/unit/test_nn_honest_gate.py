"""
tests/unit/test_nn_honest_gate.py — why the live NN trainer never promoted a
model, and the safety of the fix.

Root cause (measured on the live log and on synthetic gold-like M5 data): the
class-weighted, false-positive-penalised loss biased every model towards NOISE,
and one scalar temperature per member cannot remove a class bias, so it pinned
at its 10.0 clip and flattened every probability to ~1/3. No bar ever reached a
trade threshold, so every challenger was "unmeasured".

The fix calibrates the ensemble with a per-class bias, counts trades one
position at a time, picks the threshold on validation, and promotes only a model
whose holdout edge is positive with confidence, better than the naive
always-one-direction baseline and shows timing skill beyond its direction.

Every test here is small and seeded. Skipped when torch is not installed.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from core.ai.nn_features import (  # noqa: E402
    LONG, NOISE, SHORT, FeatureScaler, make_labels, make_windows, window_end_indices,
)
from core.ai.deep_layout import build_model_features  # noqa: E402
from core.ai.nn_model import (  # noqa: E402
    CAL_KIND_TEMPERATURE, LCB_Z, TrainConfig, _train_val_split, apply_calibration,
    baseline_metrics, edge_report, ensemble_proba, evaluate_artifact, fit_artifact,
    fit_calibration_map, fit_temperature, load_artifact, members_from_artifact,
    save_artifact, select_threshold, softmax_np, threshold_grid, trade_metrics,
)
from core.ai.nn_predictor import ModelPredictor  # noqa: E402
from core.ai.nn_trainer import ContinuousTrainer, should_promote  # noqa: E402
from tests.unit.test_nn_features import random_walk  # noqa: E402

# Small and fast; the decisions do not depend on the network's size.
CFG = TrainConfig(window=8, hidden=16, epochs=6, batch=64, horizon=6, patience=3,
                  n_models=1, threshold=0.5, lr=5e-3)


def _frame(close, rng, wick=None):
    n = len(close)
    open_ = np.concatenate([[close[0]], close[:-1]])
    if wick is None:
        up, dn = np.abs(rng.normal(0, 0.4, n)), np.abs(rng.normal(0, 0.4, n))
    else:
        up = dn = np.full(n, wick)
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + up,
                         "low": np.minimum(open_, close) - dn, "close": close,
                         "tick_volume": rng.integers(50, 500, n)}, index=idx)


def regime_bars(n=3000, seed=1, drift=0.6, dwell=60):
    """Strongly predictable: the drift flips sign every ~``dwell`` bars, and the
    last few returns reveal which regime is on. Symmetric, so the naive
    always-long / always-short baselines are near zero."""
    rng = np.random.default_rng(seed)
    state, regime = 1.0, np.empty(n)
    for i in range(n):
        if rng.random() < 1.0 / dwell:
            state = -state
        regime[i] = state
    return _frame(2000.0 + np.cumsum(drift * regime + rng.normal(0, 1.0, n)), rng)


def flat_atr_trend(n=3000, seed=1, p_up=0.56):
    """A steady up-drift with NO timing information: +-1 steps (P(up)=p_up) and
    constant wicks, so the ATR and the barriers are constant and no feature can
    tell a better entry from a worse one. Always-long wins; nothing beats it."""
    rng = np.random.default_rng(seed)
    steps = np.where(rng.random(n) < p_up, 1.0, -1.0)
    return _frame(2000.0 + np.cumsum(steps), rng, wick=0.2)


def run_cycle(tmp_path, df, cfg=CFG, **kw):
    trainer = ContinuousTrainer(lambda: df, str(tmp_path / "nn.pt"), cfg, holdout_bars=1200,
                                min_train_bars=800, interval_s=0.0, **kw)
    return trainer.run_once()


# ── 1. calibration ───────────────────────────────────────────────────────────

def biased_logits(sd=0.2, n=4000, seed=0):
    """The live failure in miniature: NOISE is ~4 % of labels, the trade classes
    carry a weak signal, and the model's NOISE logit is inflated the way the
    class-weighted, FP-penalised loss inflated it."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0, sd, n)
    true = np.stack([np.full(n, -2.4), s, -s], axis=1)
    y = np.array([rng.choice(3, p=p) for p in softmax_np(true)])
    return true + np.array([2.6, 0.0, 0.0]), y


class TestCalibration:
    def test_the_old_scalar_temperature_pins_at_its_clip_and_flattens_everything(self):
        logits, y = biased_logits()
        t = fit_temperature(logits, y)
        p = softmax_np(logits, t)
        assert t == pytest.approx(10.0)                      # the trap seen in the live log
        assert np.maximum(p[:, LONG], p[:, SHORT]).max() < 0.40   # no trade at any threshold
        assert p[:, NOISE].mean() > 0.30                     # the class bias survives

    def test_the_bias_map_removes_the_class_bias_and_restores_confidence(self):
        logits, y = biased_logits()
        raw = softmax_np(logits)
        cal = fit_calibration_map(raw, y)
        p = apply_calibration(raw, cal)
        freq = np.bincount(y, minlength=3) / len(y)
        np.testing.assert_allclose(p.mean(axis=0), freq, atol=0.01)
        assert np.maximum(p[:, LONG], p[:, SHORT]).max() > 0.6
        # Interior solution: no clip exists, and the fitted scale is ordinary.
        assert cal["fitted"] and 0.2 < cal["scale"][0] < 5.0
        assert all(abs(b) < 10 for b in cal["bias"])

        def nll(q):
            return -np.log(q[np.arange(len(y)), y]).mean()
        assert nll(p) < nll(softmax_np(logits, fit_temperature(logits, y))) - 0.2

    def test_per_class_vector_scaling_is_available_and_interior_too(self):
        logits, y = biased_logits(sd=0.6)
        cal = fit_calibration_map(softmax_np(logits), y, per_class_scale=True)
        assert cal["fit"] == "vector" and all(0.05 < a < 20 for a in cal["scale"])
        p = apply_calibration(softmax_np(logits), cal)
        np.testing.assert_allclose(p.mean(axis=0), np.bincount(y, minlength=3) / len(y),
                                   atol=0.01)

    def test_too_few_rows_give_the_identity(self):
        cal = fit_calibration_map(np.full((5, 3), 1 / 3), np.array([0, 1, 2, 1, 2]))
        assert cal["fitted"] is False
        p = np.array([[0.2, 0.5, 0.3]])
        np.testing.assert_allclose(apply_calibration(p, cal), p, atol=1e-12)

    def test_an_unknown_calibration_kind_is_refused(self):
        with pytest.raises(ValueError):
            apply_calibration(np.full((1, 3), 1 / 3), {"kind": "isotonic"})

    def test_a_real_artifact_is_calibrated_on_validation_and_generalises(self):
        df = regime_bars(n=4000, seed=7)
        art = fit_artifact(df.iloc[:3000], CFG)
        cal = art["calibration"]
        assert cal["fitted"] and 0.2 < cal["scale"][0] < 5.0
        assert all(m["temperature"] == 1.0 for m in art["members"])   # no per-member trap
        # Validation rows: mean calibrated probability == label frequency.
        feats = build_model_features(df.iloc[:3000])
        labels = make_labels(df.iloc[:3000], CFG.barrier_mult, CFG.horizon)
        x, y, ends = make_windows(FeatureScaler.from_dict(art["scaler"]).transform(feats),
                                  labels, CFG.window)
        _, cut = _train_val_split(ends, CFG)
        members = members_from_artifact(art)
        p_val = ensemble_proba(members, x[cut:], cal)
        np.testing.assert_allclose(p_val.mean(axis=0),
                                   np.bincount(y[cut:], minlength=3) / len(y[cut:]), atol=0.02)
        # Later, unseen bars of the same process: still close.
        feats = build_model_features(df)
        labels = make_labels(df, CFG.barrier_mult, CFG.horizon)
        x, y, ends = make_windows(FeatureScaler.from_dict(art["scaler"]).transform(feats),
                                  labels, CFG.window)
        later = ends >= 3000
        p_new = ensemble_proba(members, x[later], cal)
        np.testing.assert_allclose(p_new.mean(axis=0),
                                   np.bincount(y[later], minlength=3) / later.sum(), atol=0.06)


# ── 2. honest trade counting ─────────────────────────────────────────────────

def calls(direction, conf, n):
    p = np.full((n, 3), (1.0 - conf) / 2)
    p[:, direction] = conf
    return p


class TestNonOverlappingTrades:
    def test_consecutive_signals_are_one_position_per_horizon(self):
        proba = calls(LONG, 0.8, 30)
        y = np.full(30, LONG)
        m = trade_metrics(proba, y, 0.6, min_trades=1, ends=np.arange(100, 130), horizon=6)
        assert m["signals"] == 30
        assert m["trades"] == 5                     # bars 100, 106, 112, 118, 124
        assert m["wins"] == 5 and m["edge"] == pytest.approx(1.0)

    def test_the_next_trade_is_allowed_exactly_at_end_plus_horizon(self):
        proba = calls(SHORT, 0.9, 4)
        ends = np.array([10, 15, 16, 30])
        m = trade_metrics(proba, np.full(4, SHORT), 0.5, min_trades=1, ends=ends, horizon=6)
        assert m["trades"] == 3                     # 10, 16 (=10+6), 30; 15 overlaps 10

    def test_rows_are_taken_in_time_order_even_if_given_unsorted(self):
        proba = calls(LONG, 0.9, 4)
        y = np.array([SHORT, LONG, LONG, LONG])     # the earliest bar (ends=1) is the loss
        m = trade_metrics(proba, y, 0.5, min_trades=1, ends=np.array([1, 0, 9, 3]), horizon=4)
        # Time order 0(win), 1/3 overlap, 9(win): two trades, no loss.
        assert m["trades"] == 2 and m["losses"] == 0

    def test_many_signals_but_few_positions_leave_the_edge_unmeasured(self):
        n = 6 * 29                                  # 174 signal bars -> 29 positions
        proba = calls(LONG, 0.9, n)
        m = trade_metrics(proba, np.full(n, LONG), 0.5, ends=np.arange(n), horizon=6)
        assert m["signals"] == n and m["trades"] == 29
        assert m["edge"] is None and m["edge_lcb"] is None     # 29 < 30: None, never 0
        assert m["precision"] == 1.0

    def test_r_multiples_and_the_lower_bound_are_hand_computed(self):
        # 40 positions: 22 wins, 12 losses, 6 timeouts.
        y = np.array([LONG] * 22 + [SHORT] * 12 + [NOISE] * 6)
        proba = calls(LONG, 0.8, 40)
        m = trade_metrics(proba, y, 0.7, ends=np.arange(40) * 10, horizon=6)
        r = np.array([1.0] * 22 + [-1.0] * 12 + [0.0] * 6)
        assert m["trades"] == 40 and m["wins"] == 22 and m["losses"] == 12
        assert m["edge"] == pytest.approx(r.mean())
        assert m["edge_lcb"] == pytest.approx(r.mean() - LCB_Z * r.std(ddof=1) / math.sqrt(40))
        assert m["precision"] == pytest.approx(22 / 40)

    def test_a_per_row_threshold_is_honoured(self):
        proba = calls(LONG, 0.7, 4)
        m = trade_metrics(proba, np.full(4, LONG), np.array([0.6, 0.8, 0.6, 0.8]),
                          min_trades=1, ends=np.arange(4) * 10, horizon=6)
        assert m["signals"] == 2 and m["trades"] == 2

    def test_without_ends_every_signal_counts_as_before(self):
        proba = calls(LONG, 0.9, 10)
        assert trade_metrics(proba, np.full(10, LONG), 0.5, min_trades=1)["trades"] == 10


class TestBaselines:
    def test_always_long_on_rising_labels_and_its_mirror(self):
        y = np.array([LONG, LONG, SHORT, NOISE] * 30)            # 120 rows
        b = baseline_metrics(y, np.arange(120), horizon=1)
        assert b["baseline_long_edge"] == pytest.approx((60 - 30) / 120)
        assert b["baseline_short_edge"] == pytest.approx(-(60 - 30) / 120)
        assert b["baseline_edge"] == pytest.approx(0.25) and b["baseline_direction"] == "long"

    def test_too_few_rows_leave_the_baseline_unmeasured(self):
        b = baseline_metrics(np.full(100, LONG), np.arange(100), horizon=6)   # 17 positions
        assert b["baseline_edge"] is None

    def test_an_always_long_model_has_no_excess_over_the_naive_baseline(self):
        rng = np.random.default_rng(3)
        y = np.where(rng.random(600) < 0.65, LONG, SHORT)
        proba = calls(LONG, 0.9, 600)
        rep = edge_report(proba, y, 0.5, np.arange(600), horizon=6)
        assert rep["edge"] == pytest.approx(rep["baseline_edge"])   # the same positions
        assert rep["excess"] == pytest.approx(0.0, abs=1e-12)
        ok, why = should_promote(None, rep)
        assert ok is False and "naive" in why


# ── 3. the threshold is chosen on validation, never below the minimum ────────

class TestThresholdSelection:
    def test_the_default_floor_is_reachable_by_a_calibrated_model(self):
        """Regression: the default search floor must sit below what a calibrated
        3-class model can actually reach.

        It used to be 0.72. A calibrated model on noisy intraday bars tops out
        near 0.80, so the whole grid [0.72, 0.90] scored 0 non-overlapping
        trades, every edge came back "unmeasured", and no model was ever
        promoted on any symbol. The floor is a search bound, not a safety
        setting -- the >=MIN_TRADES_FOR_EDGE and positive-lower-bound gates are
        what keep an unproven model from voting.
        """
        default = TrainConfig()
        assert default.threshold <= 0.60, (
            "the default search floor must leave room below a calibrated "
            "model's reachable confidence, or the edge can never be measured"
        )
        grid = threshold_grid(default)
        assert grid[0] <= 0.60
        # The band a real model actually occupies has to be inside the grid.
        assert grid.min() < 0.75 < grid.max()

    def test_an_unreachable_floor_measures_nothing(self):
        """Why the floor matters: with calls capped at 0.70 confidence, a 0.72
        floor yields no trades at all, which is exactly 'edge unmeasured'."""
        proba = calls(LONG, 0.70, 200)
        y = np.full(200, LONG)
        m = trade_metrics(proba, y, 0.72, ends=np.arange(200), horizon=1)
        assert m.get("trades", 0) == 0
        # The same model measured at a reachable floor does produce trades.
        m2 = trade_metrics(proba, y, 0.50, ends=np.arange(200), horizon=1)
        assert m2.get("trades", 0) > 0

    def test_the_grid_starts_at_the_configured_minimum_and_never_below_040(self):
        assert threshold_grid(TrainConfig(threshold=0.55))[0] == pytest.approx(0.55)
        assert threshold_grid(TrainConfig(threshold=0.10))[0] == pytest.approx(0.40)
        assert threshold_grid(TrainConfig(threshold=0.72))[-1] == pytest.approx(0.90)

    def test_it_picks_the_confident_band_that_has_an_edge(self):
        rng = np.random.default_rng(0)
        n = 3000
        conf = rng.uniform(0.45, 0.85, n)
        # Above 0.7 the call is right 80 % of the time; below it is a coin flip.
        right = rng.random(n) < np.where(conf >= 0.7, 0.8, 0.5)
        y = np.where(right, LONG, SHORT)
        proba = np.stack([np.full(n, 0.0), conf, 1 - conf], axis=1)
        sel = select_threshold(proba, y, np.arange(n), TrainConfig(threshold=0.5, horizon=1))
        assert sel["source"] == "validation"
        assert sel["threshold"] >= 0.68                  # inside the skilled band, not the coin flips
        best = sel["validation"]["edge_lcb"]
        assert best > 0
        for t in (0.5, 0.6, 0.66):                       # any lower threshold has a worse bound
            assert trade_metrics(proba, y, t, ends=np.arange(n), horizon=1)["edge_lcb"] < best

    def test_nothing_qualifies_means_the_minimum(self):
        proba = calls(LONG, 0.9, 100)
        sel = select_threshold(proba, np.full(100, SHORT), np.arange(100),
                               TrainConfig(threshold=0.6, horizon=1))
        assert sel["threshold"] == pytest.approx(0.6) and sel["source"].startswith("fallback")

    def test_search_off_uses_the_configured_threshold(self):
        sel = select_threshold(calls(LONG, 0.9, 50), np.full(50, LONG), np.arange(50),
                               TrainConfig(threshold=0.72, threshold_search=False))
        assert sel["threshold"] == 0.72 and sel["source"] == "config"


# ── 4. the scaler sees the training rows only ────────────────────────────────

def test_the_scaler_is_fitted_on_the_training_windows_only():
    df = regime_bars(n=1500, seed=4)
    art = fit_artifact(df, CFG)
    feats = build_model_features(df)
    ends = window_end_indices(feats, make_labels(df, CFG.barrier_mult, CFG.horizon), CFG.window)
    keep, cut = _train_val_split(ends, CFG)
    train_ends = ends[keep]
    expected = FeatureScaler().fit(feats[train_ends.min() - CFG.window + 1:train_ends.max() + 1])
    np.testing.assert_allclose(art["scaler"]["mean"], expected.mean, atol=1e-12)
    with_val = FeatureScaler().fit(feats[:ends[-1] + 1])
    assert not np.allclose(art["scaler"]["mean"], with_val.mean)


# ── 5. end to end: the real trainer, the real gate, the real predictor ───────

class TestEndToEnd:
    def test_a_random_walk_is_never_promoted(self, tmp_path):
        res = run_cycle(tmp_path, random_walk(n=3000, seed=1))
        assert res["action"] == "kept"
        assert not (tmp_path / "nn.pt").exists()

    def test_a_strongly_predictable_series_is_promoted_and_votes_at_its_threshold(self, tmp_path):
        # Two stages: a passing challenger first goes on probation (no vote),
        # and is promoted only after later bars confirm it. The longer series
        # shares its first 3000 bars with regime_bars(n=3000, seed=1).
        df = regime_bars(n=4200, seed=1)
        screen = run_cycle(tmp_path, df.iloc[:3000])
        assert screen["action"] == "candidate", screen["reason"]
        assert not (tmp_path / "nn.pt").exists()          # a probation model cannot vote
        res = run_cycle(tmp_path, df)
        assert res["action"] == "promoted", res["reason"]
        art = load_artifact(str(tmp_path / "nn.pt"))
        oos = art["oos"]
        assert oos["source"] == "confirmation" and oos["confirmation"] == 1
        assert oos["trades"] >= 30
        assert oos["edge_lcb"] > 0 and oos["edge"] > oos["baseline_edge"] and oos["excess_lcb"] > 0
        stored = art["threshold"]
        assert stored >= CFG.threshold and oos["threshold"] == stored
        assert art["threshold_selection"]["source"] == "validation"

        p = ModelPredictor(str(tmp_path / "nn.pt"), min_confidence=0.0, min_oos_edge=0.0)
        assert p.threshold == pytest.approx(stored)          # the edge was measured there
        sigs = [p.predict_signal(df.iloc[:end]) for end in range(2600, 3000, 5)]
        votes = [s for s in sigs if s.direction != 0]
        below = [s for s in sigs if s.direction == 0 and "below" in s.reason]
        assert votes and all(s.confidence >= stored for s in votes)
        assert all(s.confidence < stored for s in below)
        # The operator's minimum still wins when it is stricter.
        strict = ModelPredictor(str(tmp_path / "nn.pt"), min_confidence=0.99, min_oos_edge=0.0)
        assert strict.predict_signal(df).direction == 0
        assert strict.threshold == pytest.approx(0.99)

    def test_a_trend_rider_with_no_timing_skill_is_refused(self, tmp_path):
        res = run_cycle(tmp_path, flat_atr_trend(n=3000, seed=1))
        assert res["action"] == "kept"
        c = res["challenger"]
        assert c["baseline_direction"] == "long" and c["baseline_edge"] > 0.15
        assert ("naive" in res["reason"] or "timing skill" in res["reason"]
                or "unmeasured" in res["reason"]), res["reason"]
        assert not (tmp_path / "nn.pt").exists()

    def test_a_floor_above_the_lower_bound_keeps_even_a_good_model_out(self, tmp_path):
        res = run_cycle(tmp_path, regime_bars(n=3000, seed=1), floor=0.95)
        assert res["action"] == "kept" and "not confidently above the floor" in res["reason"]


# ── 6. backward compatibility and the predictor's gates ──────────────────────

LEGACY_CFG = TrainConfig(window=8, hidden=8, epochs=2, batch=64, horizon=6, patience=2,
                         n_models=2, class_weighting=True, fp_penalty=2.0,
                         calibration=CAL_KIND_TEMPERATURE, threshold_search=False)
NEW_KEYS = ("calibration", "threshold", "threshold_selection", "train_label_freq",
            "baseline_direction")
NEW_CFG_KEYS = ("class_weighting", "calibration", "threshold_max", "threshold_step",
                "threshold_search", "lcb_z")


@pytest.fixture(scope="module")
def old_style_path(tmp_path_factory):
    """An artifact exactly as the old code saved it: per-member temperatures,
    no calibration map, no stored threshold, an oos block with only ``edge``."""
    art = fit_artifact(random_walk(n=1500, seed=3), LEGACY_CFG)
    for k in NEW_KEYS:
        art.pop(k, None)
    for k in NEW_CFG_KEYS:
        art["config"].pop(k, None)
    art["oos"] = {"edge": 0.10, "trades": 40}
    path = str(tmp_path_factory.mktemp("legacy") / "nn.pt")
    save_artifact(path, art)
    return path


class TestBackwardCompatibility:
    def test_an_old_artifact_predicts_exactly_as_before(self, old_style_path):
        df = random_walk(n=1500, seed=3)
        art = load_artifact(old_style_path)
        members = members_from_artifact(art)
        assert any(m["temperature"] != 1.0 for m in members)     # per-member temperatures kept
        p = ModelPredictor(old_style_path, min_confidence=0.0, min_oos_edge=0.0)
        sig = p.predict_signal(df)
        # The old formula: mean of softmax(logits / T_member), no calibration map.
        from core.ai.deep_layout import latest_model_window
        w = FeatureScaler.from_dict(art["scaler"]).transform(
            latest_model_window(df.tail(8 + 400), 8))[None]
        from core.ai.nn_model import logits_for
        old = np.mean([softmax_np(logits_for(m["model"], w), m["temperature"])
                       for m in members], axis=0)[0]
        np.testing.assert_allclose(sig.probs, old, atol=1e-9)
        assert p.threshold == 0.0                                # the configured min_confidence
        assert sig.direction != 0 or sig.reason == "model favours no trade"

    def test_an_old_artifact_keeps_only_the_old_edge_gate(self, old_style_path):
        df = random_walk(n=1500, seed=3)
        # Passes with a point edge only (no lower bound, no baseline): as before.
        assert "edge" not in ModelPredictor(old_style_path, min_confidence=0.0,
                                            min_oos_edge=0.05).predict_signal(df).reason
        sig = ModelPredictor(old_style_path, min_confidence=0.0,
                             min_oos_edge=0.2).predict_signal(df)
        assert sig.direction == 0 and "below floor" in sig.reason

    def test_an_old_artifact_is_scored_at_its_config_threshold(self, old_style_path):
        res = evaluate_artifact(load_artifact(old_style_path), random_walk(n=1500, seed=3), 1000)
        assert res["threshold"] == pytest.approx(LEGACY_CFG.threshold)


@pytest.fixture(scope="module")
def new_style(tmp_path_factory):
    df = random_walk(n=1500, seed=3)
    art = fit_artifact(df, CFG)
    return df, art, tmp_path_factory.mktemp("new")


def predictor_with(directory, art, oos, name, **kw):
    a = dict(art)
    a["oos"] = oos
    a["threshold"] = 0.0
    path = str(directory / name)
    save_artifact(path, a)
    return ModelPredictor(path, **{"min_confidence": 0.0, "min_oos_edge": 0.05, **kw})


GOOD = {"edge": 0.30, "edge_lcb": 0.12, "baseline_edge": 0.02, "excess_lcb": 0.08,
        "baseline_direction": "long"}


class TestPredictorGates:
    @pytest.mark.parametrize("change, words", [
        ({"edge_lcb": None}, "no confidence bound"),
        ({"edge_lcb": 0.04}, "not confidently above the floor"),
        ({"baseline_edge": None}, "no naive baseline"),
        ({"baseline_edge": 0.31}, "no better than the naive"),
        ({"excess_lcb": None}, "no timing skill"),
        ({"excess_lcb": -0.01}, "no timing skill"),
        ({"edge": None}, "never measured"),
    ])
    def test_a_new_artifact_with_a_weak_record_does_not_vote(self, new_style, change, words):
        df, art, d = new_style
        oos = {**GOOD, **change}
        if "edge_lcb" not in change and change.get("edge", 1) is None:
            oos["edge_lcb"] = None
        sig = predictor_with(d, art, oos, f"weak_{len(words)}_{words[:6]}.pt").predict_signal(df)
        assert sig.direction == 0 and words in sig.reason, sig.reason

    def test_a_missing_key_on_a_new_artifact_refuses(self, new_style):
        df, art, d = new_style
        sig = predictor_with(d, art, {"edge": 0.3}, "edge_only.pt").predict_signal(df)
        assert sig.direction == 0 and "no confidence bound" in sig.reason

    def test_a_full_record_votes_and_uses_the_calibration(self, new_style):
        df, art, d = new_style
        p = predictor_with(d, art, GOOD, "good.pt")
        sig = p.predict_signal(df)
        assert sig.direction != 0 or sig.reason == "model favours no trade"
        assert p._calibration == art["calibration"]

    def test_the_stored_threshold_is_used_but_never_below_min_confidence(self, new_style):
        df, art, d = new_style
        a = dict(art, oos=GOOD, threshold=0.66)
        path = str(d / "thr.pt")
        save_artifact(path, a)
        assert ModelPredictor(path, min_confidence=0.55).threshold == pytest.approx(0.66)
        assert ModelPredictor(path, min_confidence=0.70).threshold == pytest.approx(0.70)
        p = ModelPredictor(path, min_confidence=0.55, min_oos_edge=0.05)
        sig = p.predict_signal(df)
        assert sig.direction == 0 or sig.confidence >= 0.66
        if "below" in sig.reason:
            assert "0.66" in sig.reason


# ── 7. retraining after a deploy, and the floor in the settings ──────────────

def test_a_new_floor_or_recipe_changes_the_settings_key(tmp_path):
    df = random_walk(n=2000, seed=2)
    a = ContinuousTrainer(lambda: df, str(tmp_path / "nn.pt"), CFG, floor=0.0)
    b = ContinuousTrainer(lambda: df, str(tmp_path / "nn.pt"), CFG, floor=0.05)
    assert a.settings_key != b.settings_key
    key = json.loads(a.settings_key)
    assert key["cfg"]["calibration"] == "bias" and key["cfg"]["class_weighting"] is False
    assert key["cfg"]["fp_penalty"] == 0.0 and key["floor"] == 0.0


def test_history_written_by_the_old_recipe_does_not_delay_the_first_new_cycle(tmp_path):
    """After the deploy, the old 'kept - unmeasured' verdicts were reached with the
    old loss/calibration; they must not make the trainer wait ~a day."""
    df = random_walk(n=2000, seed=2)
    old_key = json.dumps({"cfg": {"fp_penalty": 2.0}, "holdout_bars": 1200},
                         sort_keys=True, default=str)
    with open(str(tmp_path / "nn.pt") + ".history.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"action": "kept", "reason": "challenger edge unmeasured",
                             "data_end": str(df.index[-1]), "settings": old_key}) + "\n")
    t = ContinuousTrainer(lambda: df, str(tmp_path / "nn.pt"), CFG, holdout_bars=1200)
    assert t._last_data_end is None
    assert os.path.exists(str(tmp_path / "nn.pt") + ".history.jsonl")
