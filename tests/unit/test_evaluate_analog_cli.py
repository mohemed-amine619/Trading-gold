"""
tests/unit/test_evaluate_analog_cli.py — scripts/evaluate_analog.py on synthetic
bars: a repeating market must be proven AND hold after the proof; a random walk
must never be proven.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.strategy.analog import AnalogConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ev():
    path = REPO_ROOT / "scripts" / "evaluate_analog.py"
    spec = importlib.util.spec_from_file_location("evaluate_analog", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_analog"] = module
    spec.loader.exec_module(module)
    return module


def frame(close):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2026-01-05", periods=len(close), freq="5min")
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
                        index=idx)


def sine(n, seed=0):
    rng = np.random.default_rng(seed)
    return frame(2000 + 10 * np.sin(2 * np.pi * np.arange(n) / 60) + rng.normal(0, 0.2, n))


def walk(n, seed=0):
    return frame(2000 + np.cumsum(np.random.default_rng(seed).normal(0, 1.0, n)))


CFG = AnalogConfig()


def test_a_repeating_market_is_proven_and_the_edge_holds_after_the_proof(ev):
    r = ev.evaluate(sine(1800), CFG, window=900, step=100, forward=200)
    assert r["windows"] >= 4
    assert r["proven_share"] == 1.0
    assert r["forward_calls_proven"] > 100
    assert r["forward_edge_proven"] > 0.8


def test_a_random_walk_never_proves_itself(ev):
    for seed in range(4):
        r = ev.evaluate(walk(1800, seed), CFG, window=900, step=100, forward=200)
        assert r["proven_windows"] == 0 and r["forward_edge_proven"] is None


def test_always_on_is_reported_for_comparison_and_is_near_zero_on_noise(ev):
    r = ev.evaluate(walk(1800, 1), CFG, window=900, step=100, forward=200)
    assert r["forward_calls_always_on"] > 0
    assert abs(r["forward_edge_always_on"]) < 0.35


def test_forward_scoring_uses_no_bars_from_the_proof_window_as_future(ev):
    df = sine(1500)
    engine = ev.AnalogEngine(CFG)
    vectors, outcomes, _ = engine._prepare(df)
    # scoring bars 900..1000 must equal scoring them from histories that end at each bar
    w, l, c = ev.forward_score(engine, vectors, outcomes, 900, 1000)
    w2 = l2 = c2 = 0
    for t in range(900, 1000):
        sig = engine.predict_at(df, t)
        if sig.abstained or sig.direction == 0:
            continue
        c2 += 1
        res = outcomes[t]
        if np.isnan(res):
            continue
        if (sig.direction == 1) == (res == 1.0):
            w2 += 1
        else:
            l2 += 1
    assert (w, l, c) == (w2, l2, c2)


def test_cli_reports_and_gives_a_verdict(ev, tmp_path, capsys):
    df = sine(1800)
    df["tick_volume"] = 100.0
    df.index.name = "time"
    csv = tmp_path / "bars.csv"
    df.to_csv(csv)
    assert ev.main(["--csv", str(csv), "--window", "900", "--forward", "200"]) == 0
    out = capsys.readouterr().out
    assert "proved itself in" in out and "forward edge, proven" in out and "verdict:" in out
    assert "held after proof" in out


def test_cli_says_so_when_a_market_never_proves_itself(ev, tmp_path, capsys):
    df = walk(1800, 2)
    df.index.name = "time"
    csv = tmp_path / "bars.csv"
    df.to_csv(csv)
    assert ev.main(["--csv", str(csv), "--window", "900", "--forward", "200"]) == 0
    assert "never proved itself" in capsys.readouterr().out


def test_cli_refuses_too_little_data(ev, tmp_path, capsys):
    df = sine(300)
    df.index.name = "time"
    csv = tmp_path / "bars.csv"
    df.to_csv(csv)
    assert ev.main(["--csv", str(csv)]) == 1
    assert "not enough data" in capsys.readouterr().out
