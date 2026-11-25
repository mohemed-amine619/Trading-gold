"""scripts/evaluate_strategy_features.py runs end to end on a small synthetic CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from tests.unit.test_strategy_library import market

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_strategy_features.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_strategy_features", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runs_and_reports(tmp_path, capsys):
    df = market(3200, seed=21)
    df["spread"] = 160.0
    path = tmp_path / "bars.csv"
    df.rename_axis("time").reset_index().to_csv(path, index=False)
    out = tmp_path / "table.csv"
    mod = _load()
    rc = mod.main(["--csv", str(path), "--holdout-days", "8", "--folds", "2", "--model", "logit",
                   "--strategies", "classic,smc", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "noise check" in text and "model comparison" in text
    assert "base+strategy(compact)" in text
    assert out.exists()


def test_refuses_short_data(tmp_path, capsys):
    df = market(800, seed=1)
    path = tmp_path / "bars.csv"
    df.rename_axis("time").reset_index().to_csv(path, index=False)
    assert _load().main(["--csv", str(path), "--no-models"]) == 1
