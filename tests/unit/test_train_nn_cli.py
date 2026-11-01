"""
tests/unit/test_train_nn_cli.py — scripts/train_nn.py on synthetic exported bars.

Skipped when torch is not installed. No MT5 and no data/ on disk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from core.ai.nn_model import load_artifact  # noqa: E402
from core.ai.nn_predictor import ModelPredictor  # noqa: E402
from tests.unit.test_nn_features import random_walk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGS = ["--window", "8", "--horizon", "6", "--epochs", "2",
        "--train-bars", "600", "--test-bars", "200"]


@pytest.fixture(scope="module")
def cli():
    path = REPO_ROOT / "scripts" / "train_nn.py"
    spec = importlib.util.spec_from_file_location("train_nn", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_nn"] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path, n):
    df = random_walk(n=n, seed=11)
    df["spread"] = 20.0
    df.index.name = "time"
    df.to_csv(path)


def test_trains_saves_and_reports(cli, tmp_path, capsys):
    csv = tmp_path / "bars.csv"
    out = tmp_path / "models" / "nn.pt"
    write_csv(csv, 1500)

    assert cli.main(["--csv", str(csv), "--out", str(out)] + ARGS) == 0
    assert out.exists()
    art = load_artifact(str(out))
    assert art["oos"]["n_folds"] >= 3
    report = capsys.readouterr().out
    assert "walk-forward folds" in report and "saved" in report


def test_a_model_trained_on_noise_is_refused_by_the_predictor(cli, tmp_path):
    csv = tmp_path / "bars.csv"
    out = tmp_path / "nn.pt"
    write_csv(csv, 1500)
    cli.main(["--csv", str(csv), "--out", str(out)] + ARGS)

    sig = ModelPredictor(str(out), min_confidence=0.0).predict_signal(random_walk(n=1500, seed=11))
    # A random walk has no edge: the model must come out unmeasured or below the floor.
    assert sig.direction == 0
    assert "edge" in sig.reason


def test_too_little_data_is_refused_without_writing_a_file(cli, tmp_path, capsys):
    csv = tmp_path / "short.csv"
    out = tmp_path / "nn.pt"
    write_csv(csv, 300)
    assert cli.main(["--csv", str(csv), "--out", str(out)] + ARGS) == 1
    assert not out.exists()
    assert "not enough data" in capsys.readouterr().out
