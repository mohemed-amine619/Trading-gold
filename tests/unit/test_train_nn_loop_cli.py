"""
tests/unit/test_train_nn_loop_cli.py — scripts/train_nn_loop.py with an injected
bar source (no MT5). Skipped when torch is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tests.unit.test_nn_features import random_walk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ARGS = ["--window", "8", "--horizon", "6", "--epochs", "2", "--models", "2",
        "--holdout-bars", "600", "--min-train-bars", "800", "--interval-min", "0"]


@pytest.fixture(scope="module")
def loop():
    path = REPO_ROOT / "scripts" / "train_nn_loop.py"
    spec = importlib.util.spec_from_file_location("train_nn_loop", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_nn_loop"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bars():
    return random_walk(n=2500, seed=31)


def test_once_runs_a_cycle_and_reports(loop, tmp_path, bars, capsys):
    out = tmp_path / "nn.pt"
    code = loop.main(["--out", str(out), "--once"] + ARGS, fetch_bars=lambda: bars)
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.split(":")[0] in ("kept", "promoted", "skipped")
    assert (tmp_path / "nn.pt.history.jsonl").exists()


def test_once_returns_failure_when_the_bars_cannot_be_fetched(loop, tmp_path, capsys):
    def broken():
        raise RuntimeError("terminal not connected")

    code = loop.main(["--out", str(tmp_path / "nn.pt"), "--once"] + ARGS, fetch_bars=broken)
    assert code == 1
    assert "terminal not connected" in capsys.readouterr().out


def test_the_loop_runs_until_stopped(loop, tmp_path, bars):
    stop = threading.Event()
    calls = []

    def fetch():
        calls.append(1)
        stop.set()                      # one cycle, then ask it to stop
        return bars

    code = loop.main(["--out", str(tmp_path / "nn.pt")] + ARGS, fetch_bars=fetch, stop=stop)
    assert code == 0 and len(calls) == 1


def test_a_failing_fetch_does_not_end_the_loop(loop, tmp_path, bars):
    stop = threading.Event()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("disconnected")
        stop.set()
        return bars

    code = loop.main(["--out", str(tmp_path / "nn.pt")] + ARGS, fetch_bars=flaky, stop=stop)
    assert code == 0 and len(calls) == 2


def test_the_loop_stops_by_itself_when_its_parent_process_is_gone(loop, tmp_path, bars, monkeypatch):
    import subprocess
    monkeypatch.setattr(loop, "PARENT_POLL_S", 0.05)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()                                    # a real, finished process id
    stop = threading.Event()
    args = ["--out", str(tmp_path / "nn.pt"), "--parent-pid", str(dead.pid)] + ARGS
    args[args.index("--interval-min") + 1] = "60"  # would sleep an hour if nothing stopped it
    code = loop.main(args, fetch_bars=lambda: bars, stop=stop)
    assert code == 0 and stop.is_set()


def test_a_live_parent_does_not_stop_the_loop(loop, tmp_path, bars, monkeypatch):
    import os
    monkeypatch.setattr(loop, "PARENT_POLL_S", 0.05)
    stop = threading.Event()
    calls = []

    def fetch():
        calls.append(1)
        stop.set()                                 # the test's own way to end the loop
        return bars

    args = ["--out", str(tmp_path / "nn.pt"), "--parent-pid", str(os.getpid())] + ARGS
    assert loop.main(args, fetch_bars=fetch, stop=stop) == 0 and len(calls) == 1
