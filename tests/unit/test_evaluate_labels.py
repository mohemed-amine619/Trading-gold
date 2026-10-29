"""
tests/unit/test_evaluate_labels.py — the offline label evaluator.

The evaluator reuses run.py's feature builder, labels and pipeline, so these
tests run it against the real code on synthetic bars. They check the
bookkeeping (baseline, windowing, summary) and that the label mode is honoured.
No MT5, no data/ on disk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_ai_model_rigor import _make_ohlcv, run_module  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ev():
    path = REPO_ROOT / "scripts" / "evaluate_labels.py"
    spec = importlib.util.spec_from_file_location("evaluate_labels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_labels"] = module
    spec.loader.exec_module(module)
    return module


def test_window_result_reports_edge_against_the_trivial_baseline(ev, run_module):
    result = ev.evaluate_window(run_module, _make_ohlcv(1200), "close", 5, 1.3)
    assert result is not None
    baseline = max(result["base_rate"], 1.0 - result["base_rate"])
    assert result["edge"] == pytest.approx(result["accuracy"] - baseline)
    assert result["n_val"] > 0 and result["n_train"] > result["n_val"] * 0


def test_too_little_data_gives_no_result(ev, run_module):
    assert ev.evaluate_window(run_module, _make_ohlcv(200), "close", 5, 1.3) is None


def test_walk_forward_scores_every_window_that_is_large_enough(ev, run_module):
    results = ev.walk_forward(run_module, _make_ohlcv(2600), "close", 5, 1.3,
                              window=1200, step=700)
    # Window ends at 1200, 1900, 2600: three windows, each big enough to score.
    assert len(results) == 3


def test_barrier_mode_changes_the_labels_it_trains_on(ev, run_module):
    df = _make_ohlcv(1200)
    _, y_close, _ = run_module.AIEnsemble("EVAL", ev.make_config(run_module, "close", 12, 1.3))._build_xy(df)
    _, y_barrier, _ = run_module.AIEnsemble("EVAL", ev.make_config(run_module, "barrier", 12, 1.3))._build_xy(df)
    assert not (len(y_close) == len(y_barrier) and np.array_equal(y_close, y_barrier))


def test_summary_of_known_results(ev):
    summary = ev.summarise([
        {"edge": 0.10, "auc": 0.60, "base_rate": 0.50},
        {"edge": -0.05, "auc": 0.50, "base_rate": 0.40},
    ])
    assert summary["windows"] == 2
    assert summary["mean_edge"] == pytest.approx(0.025)
    assert summary["min_edge"] == pytest.approx(-0.05)
    assert summary["share_edge_positive"] == pytest.approx(0.5)
    assert summary["mean_auc"] == pytest.approx(0.55)


def test_summary_of_nothing_is_explicit(ev):
    assert ev.summarise([]) == {"windows": 0}


def test_csv_round_trip_matches_the_bot_frame_shape(ev, tmp_path):
    df = _make_ohlcv(50)
    df.index = pd.date_range("2026-01-01", periods=len(df), freq="15min")
    df["spread"] = 10.0   # export_bars.py always writes spread; the synthetic generator does not
    path = tmp_path / "bars.csv"
    df[["open", "high", "low", "close", "tick_volume", "spread"]].rename_axis("time").to_csv(path)

    loaded = ev.load_bars(path)
    assert list(loaded.columns) == ["open", "high", "low", "close", "tick_volume", "spread"]
    assert loaded.index.name == "time"
    assert len(loaded) == len(df)
