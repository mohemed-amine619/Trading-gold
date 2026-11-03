"""
tests/unit/test_evaluate_scalp_cli.py — scripts/evaluate_scalp.py on synthetic bars:
a strategy with real edge must be proven and hold after the proof, noise must not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from core.strategy.scalp import ScalpConfig
from tests.unit.test_scalp_strategies import planted_burst_series, walk

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG = ScalpConfig(strategies=("burst", "sweep"), check_bars=800)


@pytest.fixture(scope="module")
def ev():
    path = REPO_ROOT / "scripts" / "evaluate_scalp.py"
    spec = importlib.util.spec_from_file_location("evaluate_scalp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_scalp"] = module
    spec.loader.exec_module(module)
    return module


def to_csv(df, tmp_path):
    out = df.copy()
    out.index.name = "time"
    path = tmp_path / "bars.csv"
    out.to_csv(path)
    return path


def test_a_strategy_with_real_edge_is_proven_and_the_edge_holds_after_the_proof(ev):
    df = planted_burst_series(n=3000, last_is_burst=False)
    r = ev.evaluate(df, CFG, window=1200, step=150, forward=300)["burst"]
    assert r["proven_share"] == 1.0
    assert r["forward_calls_proven"] >= 20 and r["forward_edge_proven"] > 0.5
    assert ev.verdict(r) == "edge HELD after proof"


def test_noise_is_never_proven(ev):
    for seed in range(3):
        res = ev.evaluate(walk(3000, seed), CFG, window=1200, step=150, forward=300)
        for name, r in res.items():
            assert r["proven_windows"] == 0 and r["forward_edge_proven"] is None, name
            assert ev.verdict(r) == "never proved itself"


def test_always_on_is_reported_for_comparison(ev):
    r = ev.evaluate(walk(3000, 1), CFG, window=1200, step=150, forward=300)["sweep"]
    assert r["forward_calls_always_on"] > 0 and abs(r["forward_edge_always_on"]) < 0.5


def test_verdict_wording(ev):
    base = {"proven_windows": 3, "forward_calls_proven": 50, "forward_edge_proven": 0.3}
    assert ev.verdict(base) == "edge HELD after proof"
    assert ev.verdict({**base, "forward_edge_proven": 0.0}) == "edge did NOT hold after proof (chance)"
    assert "too few calls" in ev.verdict({**base, "forward_calls_proven": 5})
    assert ev.verdict({**base, "proven_windows": 0}) == "never proved itself"


def test_the_forward_window_never_reads_into_the_proof_window(ev):
    # a proof is made on bars < end; scoring starts AT end, so the two never overlap
    df = planted_burst_series(n=2400, last_is_burst=False)
    res = ev.evaluate(df, CFG, window=1200, step=100, forward=200)["burst"]
    assert res["windows"] == len(range(1200, 2400 - 200 - CFG.horizon, 100))


def test_cli_prints_a_table_and_the_caveat(ev, tmp_path, capsys):
    path = to_csv(planted_burst_series(n=2600, last_is_burst=False), tmp_path)
    code = ev.main(["--csv", str(path), "--window", "1200", "--forward", "300", "--strategies", "burst,sweep"])
    out = capsys.readouterr().out
    assert code == 0
    assert "burst" in out and "edge HELD after proof" in out
    assert "strategies whose edge held after proof: burst" in out
    assert "one looking good by chance is expected" in out


def test_cli_says_so_when_nothing_proves_itself(ev, tmp_path, capsys):
    path = to_csv(walk(2600, 4), tmp_path)
    assert ev.main(["--csv", str(path), "--window", "1200", "--forward", "300"]) == 0
    out = capsys.readouterr().out
    assert "never proved itself" in out and "edge held after proof: none" in out


def test_cli_refuses_too_little_data(ev, tmp_path, capsys):
    path = to_csv(walk(400), tmp_path)
    assert ev.main(["--csv", str(path)]) == 1
    assert "not enough data" in capsys.readouterr().out
