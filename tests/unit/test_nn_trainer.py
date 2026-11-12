"""
tests/unit/test_nn_trainer.py — ensemble, evaluation on unseen bars, the
continuous trainer's decisions and file handling, and predictor hot reload.

The trainer's promote/keep/skip/error paths use injected fit/evaluate fakes so
they are deterministic; one test runs the real model end to end. Skipped when
torch is not installed.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from core.ai.nn_features import FeatureScaler, build_features, make_labels, make_windows  # noqa: E402
from core.ai.nn_model import (  # noqa: E402
    TrainConfig, ensemble_proba, evaluate_artifact, fit_artifact, fit_ensemble,
    load_artifact, members_from_artifact, save_artifact, softmax_np, logits_for,
)
from core.ai.nn_predictor import ModelPredictor  # noqa: E402
from core.ai.nn_trainer import ContinuousTrainer  # noqa: E402
from tests.unit.test_nn_features import random_walk  # noqa: E402

CFG = TrainConfig(window=8, hidden=8, epochs=3, batch=64, horizon=6, patience=2, n_models=2)


@pytest.fixture(scope="module")
def bars():
    return random_walk(n=2500, seed=21)


@pytest.fixture(scope="module")
def artifact(bars):
    return fit_artifact(bars.iloc[:1800], CFG)


class TestEnsemble:
    def test_builds_the_requested_number_of_members(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (240, 8, 4))
        y = rng.integers(0, 3, 240)
        members = fit_ensemble(x[:180], y[:180], x[180:], y[180:], CFG)
        assert len(members) == 2
        assert all(m["temperature"] > 0 for m in members)

    def test_probability_is_the_mean_of_the_members_and_sums_to_one(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (240, 8, 4))
        y = rng.integers(0, 3, 240)
        members = fit_ensemble(x[:180], y[:180], x[180:], y[180:], CFG)
        p = ensemble_proba(members, x[180:])
        manual = np.mean([softmax_np(logits_for(m["model"], x[180:]), m["temperature"])
                          for m in members], axis=0)
        np.testing.assert_allclose(p, manual, atol=1e-9)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_members_differ_by_seed(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, (240, 8, 4))
        y = rng.integers(0, 3, 240)
        a, b = fit_ensemble(x[:180], y[:180], x[180:], y[180:], CFG)
        assert not np.allclose(logits_for(a["model"], x[180:]), logits_for(b["model"], x[180:]))


class TestFitAndEvaluate:
    def test_artifact_records_how_far_it_trained(self, artifact, bars):
        assert artifact["trained_through"] == str(bars.index[1799])
        assert len(artifact["members"]) == CFG.n_models
        assert artifact["oos"] == {}                 # unmeasured until evaluated on unseen bars

    def test_evaluation_uses_only_windows_ending_at_or_after_start_row(self, artifact, bars):
        start = 2000
        res = evaluate_artifact(artifact, bars, start)
        # Independent count: windows ending at/after `start` with a known label.
        feats = build_features(bars).to_numpy()
        labels = make_labels(bars, CFG.barrier_mult, CFG.horizon)
        scaler = FeatureScaler.from_dict(artifact["scaler"])
        _, _, ends = make_windows(scaler.transform(feats), labels, CFG.window)
        assert res["windows"] == int((ends >= start).sum())
        assert res["windows"] <= 2500 - start - CFG.horizon     # the last horizon bars are unknown

    def test_start_beyond_the_data_gives_no_windows(self, artifact, bars):
        res = evaluate_artifact(artifact, bars, 10_000)
        assert res["windows"] == 0 and res["edge"] is None

    def test_evaluation_is_deterministic(self, artifact, bars):
        assert evaluate_artifact(artifact, bars, 2000) == evaluate_artifact(artifact, bars, 2000)

    def test_too_little_data_to_train_raises(self):
        with pytest.raises(ValueError):
            fit_artifact(random_walk(n=120, seed=1), CFG)


# ── The trainer's decisions, with injected fit / evaluate ────────────────────

class Fakes:
    """Controllable fit/evaluate. ``edge`` is what the NEXT challenger scores."""

    def __init__(self):
        self.edge = 0.10
        self.champion_edge = 0.05
        self.fit_calls = []
        self.eval_calls = []
        self.fit_error = None

    def fit(self, train_df, cfg):
        if self.fit_error:
            raise self.fit_error
        self.fit_calls.append(train_df)
        return {"marker": len(self.fit_calls), "edge": self.edge, "config": {},
                "oos": {}, "trained_through": str(train_df.index[-1])}

    def evaluate(self, art, df, start_row):
        self.eval_calls.append((art.get("marker"), len(df), start_row))
        edge = art["edge"] if art.get("marker") is not None and art["edge"] != "champion" \
            else self.champion_edge
        # Shaped like evaluate_artifact's edge_report: a measured edge comes with a
        # lower bound, a naive baseline and the timing skill beyond it (all passing
        # here, so these tests exercise the trainer's flow, not the gate itself).
        return {"windows": 500, "trades": 40, "coverage": 0.1, "precision": 0.5, "edge": edge,
                "edge_lcb": edge, "baseline_edge": None if edge is None else -0.2,
                "excess_lcb": edge, "threshold": 0.6}


def make_trainer(tmp_path, bars, fakes, **kw):
    kw.setdefault("holdout_bars", 600)
    kw.setdefault("min_new_bars", 200)
    kw.setdefault("min_train_bars", 800)
    return ContinuousTrainer(lambda: bars, str(tmp_path / "nn.pt"), CFG, tolerance=0.02,
                             interval_s=0.0, fit=fakes.fit, evaluate=fakes.evaluate, **kw)


def history(tmp_path):
    path = tmp_path / "nn.pt.history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class TestTrainerDecisions:
    def test_first_good_model_is_promoted_and_written(self, tmp_path, bars):
        fakes = Fakes()
        res = make_trainer(tmp_path, bars, fakes).run_once()
        assert res["action"] == "promoted"
        saved = load_artifact(str(tmp_path / "nn.pt"))
        assert saved["marker"] == 1
        assert saved["oos"]["source"] == "holdout" and saved["oos"]["edge"] == 0.10
        assert not (tmp_path / "nn.pt.tmp").exists()            # written atomically
        assert [h["action"] for h in history(tmp_path)] == ["promoted"]

    def test_a_better_challenger_replaces_the_champion_and_keeps_a_backup(self, tmp_path, bars):
        fakes = Fakes()
        trainer = make_trainer(tmp_path, bars, fakes)
        trainer.run_once()                                       # champion: marker 1
        fakes.edge = 0.30
        # Champion's own holdout score comes from champion_edge, via the "champion" tag.
        champ = load_artifact(str(tmp_path / "nn.pt"))
        champ["edge"] = "champion"
        save_artifact(str(tmp_path / "nn.pt"), champ)
        fakes.champion_edge = 0.05
        longer = pd.concat([bars, random_walk(n=400, seed=22).set_axis(
            pd.date_range(bars.index[-1] + pd.Timedelta(minutes=5), periods=400, freq="5min"))])
        trainer.fetch_bars = lambda: longer
        res = trainer.run_once()
        assert res["action"] == "promoted"
        assert load_artifact(str(tmp_path / "nn.pt"))["marker"] == 2
        assert load_artifact(str(tmp_path / "nn.pt.prev"))["marker"] == 1

    def test_a_challenger_that_is_not_better_enough_is_kept_out(self, tmp_path, bars):
        fakes = Fakes()
        trainer = make_trainer(tmp_path, bars, fakes)
        trainer.run_once()
        champ = load_artifact(str(tmp_path / "nn.pt"))
        champ["edge"] = "champion"
        save_artifact(str(tmp_path / "nn.pt"), champ)
        before = (tmp_path / "nn.pt").read_bytes()
        fakes.champion_edge, fakes.edge = 0.10, 0.11             # +0.01 < tolerance 0.02
        longer = pd.concat([bars, random_walk(n=400, seed=23).set_axis(
            pd.date_range(bars.index[-1] + pd.Timedelta(minutes=5), periods=400, freq="5min"))])
        trainer.fetch_bars = lambda: longer
        res = trainer.run_once()
        assert res["action"] == "kept"
        assert (tmp_path / "nn.pt").read_bytes() == before
        assert not (tmp_path / "nn.pt.prev").exists()

    def test_an_unmeasured_challenger_is_never_written(self, tmp_path, bars):
        fakes = Fakes()
        fakes.edge = None
        res = make_trainer(tmp_path, bars, fakes).run_once()
        assert res["action"] == "kept" and "unmeasured" in res["reason"]
        assert not (tmp_path / "nn.pt").exists()

    def test_no_new_bars_since_the_last_cycle_skips_training(self, tmp_path, bars):
        fakes = Fakes()
        trainer = make_trainer(tmp_path, bars, fakes)
        trainer.run_once()
        calls = len(fakes.fit_calls)
        res = trainer.run_once()                                 # same bars: nothing new
        assert res["action"] == "skipped" and "since the last training cycle" in res["reason"]
        assert len(fakes.fit_calls) == calls

    def test_a_restart_remembers_the_data_it_already_trained_on(self, tmp_path, bars):
        fakes = Fakes()
        make_trainer(tmp_path, bars, fakes).run_once()
        calls = len(fakes.fit_calls)
        restarted = make_trainer(tmp_path, bars, fakes)          # new object, same history file
        res = restarted.run_once()
        assert res["action"] == "skipped" and len(fakes.fit_calls) == calls

    def test_a_kept_cycle_also_counts_as_trained_on_that_data(self, tmp_path, bars):
        fakes = Fakes()
        fakes.edge = None                                        # challenger unmeasured: kept
        trainer = make_trainer(tmp_path, bars, fakes)
        assert trainer.run_once()["action"] == "kept"
        assert trainer.run_once()["action"] == "skipped"

    def test_an_error_does_not_mark_the_data_as_trained_on(self, tmp_path, bars):
        fakes = Fakes()
        fakes.fit_error = ValueError("boom")
        trainer = make_trainer(tmp_path, bars, fakes)
        assert trainer.run_once()["action"] == "error"
        fakes.fit_error = None
        assert trainer.run_once()["action"] == "promoted"        # retried, not skipped

    def test_a_champion_trained_on_nearly_all_the_data_has_no_fair_holdout(self, tmp_path, bars):
        fakes = Fakes()
        save_artifact(str(tmp_path / "nn.pt"),
                      {"marker": 0, "edge": 0.0, "config": {}, "oos": {},
                       "trained_through": str(bars.index[-50])})
        res = make_trainer(tmp_path, bars, fakes).run_once()
        assert res["action"] == "skipped" and "since the champion" in res["reason"]
        assert fakes.fit_calls == []

    def test_a_fetch_failure_is_reported_and_leaves_the_champion_alone(self, tmp_path, bars):
        fakes = Fakes()
        trainer = make_trainer(tmp_path, bars, fakes)
        trainer.run_once()
        before = (tmp_path / "nn.pt").read_bytes()

        def broken():
            raise RuntimeError("terminal not connected")

        trainer.fetch_bars = broken
        res = trainer.run_once()
        assert res["action"] == "error" and "terminal not connected" in res["reason"]
        assert (tmp_path / "nn.pt").read_bytes() == before

    def test_a_training_failure_is_reported_and_leaves_the_champion_alone(self, tmp_path, bars):
        fakes = Fakes()
        fakes.fit_error = ValueError("boom")
        res = make_trainer(tmp_path, bars, fakes).run_once()
        assert res["action"] == "error" and "boom" in res["reason"]
        assert not (tmp_path / "nn.pt").exists()

    def test_not_enough_bars_is_skipped(self, tmp_path):
        short = random_walk(n=400, seed=5)
        res = make_trainer(tmp_path, short, Fakes()).run_once()
        assert res["action"] == "skipped"


class TestHoldoutDiscipline:
    def test_training_stops_a_horizon_before_the_holdout(self, tmp_path, bars):
        fakes = Fakes()
        make_trainer(tmp_path, bars, fakes).run_once()
        train_df = fakes.fit_calls[0]
        holdout_start = len(bars) - 600
        assert len(train_df) == holdout_start - CFG.horizon
        assert bars.index.get_loc(train_df.index[-1]) == holdout_start - CFG.horizon - 1

    def test_champion_and_challenger_are_scored_on_the_same_bars(self, tmp_path, bars):
        fakes = Fakes()
        trainer = make_trainer(tmp_path, bars, fakes)
        trainer.run_once()
        champ = load_artifact(str(tmp_path / "nn.pt"))
        champ["edge"] = "champion"
        save_artifact(str(tmp_path / "nn.pt"), champ)
        longer = pd.concat([bars, random_walk(n=400, seed=24).set_axis(
            pd.date_range(bars.index[-1] + pd.Timedelta(minutes=5), periods=400, freq="5min"))])
        trainer.fetch_bars = lambda: longer
        fakes.eval_calls.clear()
        trainer.run_once()
        assert len(fakes.eval_calls) == 2
        (_, n1, s1), (_, n2, s2) = fakes.eval_calls
        assert (n1, s1) == (n2, s2)

    def test_the_holdout_starts_after_the_champions_training_end(self, tmp_path, bars):
        fakes = Fakes()
        trainer = make_trainer(tmp_path, bars, fakes)
        trainer.run_once()
        champ = load_artifact(str(tmp_path / "nn.pt"))
        trained_through = pd.Timestamp(champ["trained_through"])
        champ["edge"] = "champion"
        save_artifact(str(tmp_path / "nn.pt"), champ)
        longer = pd.concat([bars, random_walk(n=400, seed=25).set_axis(
            pd.date_range(bars.index[-1] + pd.Timedelta(minutes=5), periods=400, freq="5min"))])
        trainer.fetch_bars = lambda: longer
        fakes.eval_calls.clear()
        trainer.run_once()
        start_row = fakes.eval_calls[0][2]
        assert longer.index[start_row] > trained_through


class TestRunForever:
    def test_stops_on_the_event(self, tmp_path, bars):
        stop = threading.Event()
        trainer = make_trainer(tmp_path, bars, Fakes())
        original = trainer.fetch_bars

        def fetch_then_stop():
            stop.set()
            return original()

        trainer.fetch_bars = fetch_then_stop
        assert trainer.run_forever(stop) == 1

    def test_max_cycles(self, tmp_path, bars):
        assert make_trainer(tmp_path, bars, Fakes()).run_forever(threading.Event(), max_cycles=3) == 3

    def test_a_crashing_cycle_does_not_end_the_loop(self, tmp_path, bars):
        trainer = make_trainer(tmp_path, bars, Fakes())
        trainer._cycle = lambda: (_ for _ in ()).throw(RuntimeError("unexpected"))
        assert trainer.run_forever(threading.Event(), max_cycles=2) == 2


class TestRealModelEndToEnd:
    def test_real_training_on_noise_runs_and_never_writes_an_unmeasured_model(self, tmp_path, bars):
        trainer = ContinuousTrainer(lambda: bars, str(tmp_path / "nn.pt"), CFG,
                                    holdout_bars=600, min_train_bars=800, interval_s=0.0)
        res = trainer.run_once()
        assert res["action"] in ("kept", "promoted")
        if res["action"] == "kept":
            assert not (tmp_path / "nn.pt").exists()
        else:                                    # a noise model may pass by luck; it must be measured
            assert load_artifact(str(tmp_path / "nn.pt"))["oos"]["edge"] > 0
        assert history(tmp_path)[0]["holdout_bars"] == 600


class TestPredictorHotReload:
    def _save(self, path, art, edge):
        art = dict(art)
        art["oos"] = {"edge": edge, "edge_lcb": edge, "baseline_edge": -0.5, "excess_lcb": edge}
        save_artifact(str(path), art)

    def test_a_replaced_model_file_is_picked_up_without_a_restart(self, tmp_path, artifact, bars):
        path = tmp_path / "nn.pt"
        self._save(path, artifact, 0.05)
        p = ModelPredictor(str(path), min_confidence=0.0, min_oos_edge=0.0)
        assert p.predict_signal(bars.iloc[:1800]).reason not in (
            "model out-of-sample edge was never measured",) and p.status == "ready"

        self._save(path, artifact, -0.10)                       # trainer swapped in a worse file
        os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
        sig = p.predict_signal(bars.iloc[:1800])
        assert sig.direction == 0 and "below floor" in sig.reason

    def test_a_model_file_that_appears_later_is_loaded(self, tmp_path, artifact, bars):
        path = tmp_path / "later.pt"
        p = ModelPredictor(str(path), min_confidence=0.0)
        assert "no model file" in p.predict_signal(bars.iloc[:1800]).reason
        self._save(path, artifact, 0.05)
        assert p.predict_signal(bars.iloc[:1800]).direction in (-1, 0, 1)
        assert p.status == "ready"

    def test_the_ensemble_prediction_uses_every_member(self, tmp_path, artifact, bars):
        path = tmp_path / "nn.pt"
        self._save(path, artifact, 0.05)
        p = ModelPredictor(str(path), min_confidence=0.0)
        sig = p.predict_signal(bars.iloc[:1800])
        assert len(members_from_artifact(load_artifact(str(path)))) == CFG.n_models
        assert sum(sig.probs) == pytest.approx(1.0, abs=1e-6)


# ── Changing the settings retrains at once (no day-long wait) ────────────────

class TestSettingsChangeRetrains:
    """After an "edge unmeasured" verdict the operator changes the threshold or the
    holdout size. The old verdict says nothing about the new settings, so the next
    cycle must train immediately instead of waiting for min_new_bars new bars."""

    def _kept_once(self, tmp_path, bars, fakes, **kw):
        fakes.edge = None                                   # unmeasured -> kept, no file
        first = make_trainer(tmp_path, bars, fakes, **kw)
        assert first.run_once()["action"] == "kept"
        return len(fakes.fit_calls)

    def test_same_settings_still_wait_for_new_bars(self, tmp_path):
        bars, fakes = random_walk(n=2000, seed=3), Fakes()
        calls = self._kept_once(tmp_path, bars, fakes)
        again = make_trainer(tmp_path, bars, fakes)
        assert again.run_once()["action"] == "skipped"
        assert len(fakes.fit_calls) == calls

    def test_a_new_holdout_size_trains_immediately(self, tmp_path):
        bars, fakes = random_walk(n=2000, seed=3), Fakes()
        calls = self._kept_once(tmp_path, bars, fakes, holdout_bars=600)
        changed = make_trainer(tmp_path, bars, fakes, holdout_bars=700)
        assert changed.run_once()["action"] != "skipped"
        assert len(fakes.fit_calls) == calls + 1

    def test_a_new_confidence_threshold_trains_immediately(self, tmp_path):
        from dataclasses import replace
        bars, fakes = random_walk(n=2000, seed=3), Fakes()
        calls = self._kept_once(tmp_path, bars, fakes)
        changed = ContinuousTrainer(lambda: bars, str(tmp_path / "nn.pt"),
                                    replace(CFG, threshold=0.55), tolerance=0.02,
                                    interval_s=0.0, fit=fakes.fit, evaluate=fakes.evaluate,
                                    holdout_bars=600, min_new_bars=200, min_train_bars=800)
        assert changed.run_once()["action"] != "skipped"
        assert len(fakes.fit_calls) == calls + 1

    def test_every_cycle_records_its_settings(self, tmp_path):
        bars, fakes = random_walk(n=2000, seed=3), Fakes()
        self._kept_once(tmp_path, bars, fakes)
        assert all("settings" in e for e in history(tmp_path))
