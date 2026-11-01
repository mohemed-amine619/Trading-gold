"""
tests/unit/test_nn_model.py — the sequence model, its loss, calibration,
walk-forward evaluation, the saved file and the predictor.

Skipped when torch is not installed. The predictor's torch-free guards are
covered in test_nn_predictor_neutral.py, which always runs.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from core.ai.nn_features import LONG, NOISE, SHORT  # noqa: E402
from core.ai.nn_model import (  # noqa: E402
    SeqClassifier, TrainConfig, balanced_class_weights, fit_sequence_model,
    fit_temperature, fp_penalised_loss, load_artifact, logits_for, save_artifact,
    softmax_np, trade_metrics, train_final, walk_forward_evaluate,
)
from core.ai.nn_predictor import ModelPredictor  # noqa: E402
from tests.unit.test_nn_features import random_walk  # noqa: E402

SMALL = TrainConfig(window=8, hidden=8, epochs=4, batch=64, horizon=6, patience=2)


class TestLoss:
    def _logits(self, favoured):
        row = [0.0, 0.0, 0.0]
        row[favoured] = 6.0
        return torch.tensor([row])

    def test_a_confident_false_trade_costs_more_than_a_confident_no_trade(self):
        y = torch.tensor([NOISE])
        w = torch.ones(3)
        wrong_trade = fp_penalised_loss(self._logits(LONG), y, w, fp_penalty=2.0)
        right_noise = fp_penalised_loss(self._logits(NOISE), y, w, fp_penalty=2.0)
        assert wrong_trade > right_noise

    def test_the_penalty_raises_the_loss_of_a_false_trade_only(self):
        y = torch.tensor([NOISE])
        w = torch.ones(3)
        lg = self._logits(SHORT)
        assert fp_penalised_loss(lg, y, w, 3.0) > fp_penalised_loss(lg, y, w, 0.0)
        # A correct trade call adds no false-positive term.
        ok = self._logits(LONG)
        good = torch.tensor([LONG])
        assert float(fp_penalised_loss(ok, good, w, 3.0)) == pytest.approx(
            float(fp_penalised_loss(ok, good, w, 0.0)), abs=0.02)

    def test_balanced_weights_favour_the_rare_class(self):
        y = np.array([NOISE] * 80 + [LONG] * 15 + [SHORT] * 5)
        w = balanced_class_weights(y)
        assert w[SHORT] > w[LONG] > w[NOISE]
        assert float(w[NOISE]) == pytest.approx(100 / (3 * 80))

    def test_absent_class_gets_neutral_weight(self):
        w = balanced_class_weights(np.array([NOISE, NOISE, LONG]))
        assert float(w[SHORT]) == 1.0


class TestModel:
    def test_output_shape(self):
        out = SeqClassifier(5, hidden=8)(torch.zeros(7, 12, 5))
        assert out.shape == (7, 3)

    def test_it_learns_a_separable_problem(self):
        # Class is fully determined by the sign/size of the window's mean feature.
        rng = np.random.default_rng(0)
        n, w, f = 900, 8, 4
        x = rng.normal(0, 1, (n, w, f))
        m = x[:, :, 0].mean(axis=1)
        y = np.where(m > 0.35, LONG, np.where(m < -0.35, SHORT, NOISE))
        cfg = TrainConfig(window=w, hidden=16, epochs=40, batch=64, patience=10,
                          fp_penalty=0.0, lr=5e-3)
        model, info = fit_sequence_model(x[:700], y[:700], x[700:], y[700:], cfg)
        acc = (logits_for(model, x[700:]).argmax(axis=1) == y[700:]).mean()
        assert acc > 0.85, acc
        assert info["epochs_run"] >= 1

    def test_training_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (200, 8, 4))
        y = rng.integers(0, 3, 200)
        a, _ = fit_sequence_model(x[:150], y[:150], x[150:], y[150:], SMALL)
        b, _ = fit_sequence_model(x[:150], y[:150], x[150:], y[150:], SMALL)
        np.testing.assert_allclose(logits_for(a, x[150:]), logits_for(b, x[150:]), atol=1e-6)


class TestCalibration:
    def test_an_overconfident_model_gets_a_temperature_above_one(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 600)
        logits = np.eye(3)[y] * 1.0 + rng.normal(0, 1.0, (600, 3))   # informative but noisy
        overconfident = logits * 6.0
        t = fit_temperature(overconfident, y)
        assert t > 1.5

        def nll(lg, temp):
            p = softmax_np(lg, temp)
            return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

        assert nll(overconfident, t) < nll(overconfident, 1.0)

    def test_too_little_data_leaves_the_temperature_alone(self):
        assert fit_temperature(np.zeros((3, 3)), np.array([0, 1, 2])) == 1.0


class TestTradeMetrics:
    def test_hand_computed_precision(self):
        proba = np.array([
            [0.10, 0.80, 0.10],   # LONG 0.80  -> taken, truth LONG  : win
            [0.10, 0.80, 0.10],   # LONG 0.80  -> taken, truth NOISE : loss
            [0.10, 0.20, 0.70],   # SHORT 0.70 -> below 0.72, not taken
            [0.05, 0.05, 0.90],   # SHORT 0.90 -> taken, truth SHORT : win
            [0.90, 0.05, 0.05],   # NOISE      -> never a trade
            [0.10, 0.80, 0.10],   # LONG 0.80  -> taken, truth SHORT : loss
        ])
        y = np.array([LONG, NOISE, SHORT, SHORT, LONG, SHORT])
        m = trade_metrics(proba, y, threshold=0.72, min_trades=1)
        assert m["trades"] == 4                       # rows 1, 2, 4, 6
        assert m["precision"] == pytest.approx(2 / 4)  # two wins
        # +1, 0 (timed out flat), +1, -1  ->  mean 0.25 R
        assert m["edge"] == pytest.approx(0.25)
        assert m["coverage"] == pytest.approx(4 / 6)

    def test_a_flat_timeout_is_not_counted_as_a_loss(self):
        proba = np.array([[0.1, 0.8, 0.1]] * 4)
        y = np.array([NOISE] * 4)
        assert trade_metrics(proba, y, 0.72, min_trades=1)["edge"] == 0.0

    def test_too_few_trades_leave_the_edge_unmeasured(self):
        proba = np.array([[0.1, 0.9, 0.0]] * 5)
        y = np.array([LONG] * 5)                        # five wins out of five
        m = trade_metrics(proba, y, 0.72)               # default min_trades = 30
        assert m["trades"] == 5 and m["precision"] == 1.0
        assert m["edge"] is None

    def test_no_trades_means_unmeasured_not_zero_edge(self):
        m = trade_metrics(np.array([[0.9, 0.05, 0.05]]), np.array([NOISE]), 0.72)
        assert m["trades"] == 0 and m["edge"] is None and m["precision"] is None


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = random_walk(n=1500, seed=3)
    art = train_final(df, SMALL, train_bars=600, test_bars=200)
    path = str(tmp_path_factory.mktemp("nn") / "model.pt")
    save_artifact(path, art)
    return df, art, path


class TestWalkForward:
    def test_runs_and_reports_folds_without_claiming_an_edge_from_noise(self):
        df = random_walk(n=1500, seed=3)
        res = walk_forward_evaluate(df, SMALL, train_bars=600, test_bars=200)
        assert len(res["folds"]) >= 3
        for fold in res["folds"]:
            assert fold["test"][0] - fold["train"][1] == SMALL.horizon
        # A random walk has no edge to find: either no trades, or a modest one.
        assert res["edge"] is None or res["edge"] < 0.25

    def test_too_short_a_series_yields_no_folds(self):
        res = walk_forward_evaluate(random_walk(n=200), SMALL, train_bars=600, test_bars=200)
        assert res["folds"] == [] and res["edge"] is None


class TestArtifactAndPredictor:
    def test_artifact_carries_the_measured_oos_result(self, trained):
        _, art, _ = trained
        assert set(["members", "scaler", "config", "oos", "trained_through"]) <= set(art)
        assert "edge" in art["oos"] and "n_folds" in art["oos"]

    def test_file_round_trips_with_weights_only_loading(self, trained):
        _, art, path = trained
        loaded = load_artifact(path)
        assert loaded["config"]["window"] == SMALL.window
        assert loaded["members"][0]["temperature"] == pytest.approx(art["members"][0]["temperature"])

    def _predictor(self, path, edge, **kw):
        p = ModelPredictor(path, **kw)
        p._load()
        p._oos_edge = edge
        return p

    def test_probabilities_are_a_distribution_and_direction_is_valid(self, trained):
        df, _, path = trained
        sig = self._predictor(path, 0.05, min_confidence=0.0).predict_signal(df)
        assert sig.direction in (-1, 0, 1)
        assert sum(sig.probs) == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= sig.confidence <= 1.0

    def test_an_unmeasured_edge_fails_neutral(self, trained):
        df, _, path = trained
        sig = self._predictor(path, None, min_confidence=0.0).predict_signal(df)
        assert sig.direction == 0 and "never measured" in sig.reason

    def test_an_edge_below_the_floor_fails_neutral(self, trained):
        df, _, path = trained
        sig = self._predictor(path, -0.02, min_confidence=0.0, min_oos_edge=0.0).predict_signal(df)
        assert sig.direction == 0 and "below floor" in sig.reason

    def test_an_impossible_threshold_means_no_trade(self, trained):
        df, _, path = trained
        sig = self._predictor(path, 0.05, min_confidence=1.01).predict_signal(df)
        assert sig.direction == 0

    def test_too_little_history_fails_neutral(self, trained):
        df, _, path = trained
        sig = self._predictor(path, 0.05, min_confidence=0.0).predict_signal(df.iloc[:20])
        assert sig.direction == 0

    def test_a_nan_in_the_window_fails_neutral(self, trained):
        df, _, path = trained
        bad = df.copy()
        bad.iloc[-3, bad.columns.get_loc("close")] = np.nan
        sig = self._predictor(path, 0.05, min_confidence=0.0).predict_signal(bad)
        assert sig.direction == 0

    def test_prediction_is_fast_enough_for_a_bar_loop(self, trained):
        df, _, path = trained
        p = self._predictor(path, 0.05, min_confidence=0.0)
        p.predict_signal(df)                      # first call loads and warms up
        start = time.perf_counter()
        for _ in range(5):
            p.predict_signal(df)
        assert (time.perf_counter() - start) / 5 < 0.25

    def test_as_tuple_matches_the_ensemble_shape(self, trained):
        df, _, path = trained
        sig = self._predictor(path, 0.05, min_confidence=0.0).predict_signal(df)
        assert sig.as_tuple() == (sig.direction, sig.confidence)
