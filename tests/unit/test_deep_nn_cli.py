"""Deep-network trainer wiring: the default command is unchanged, dual is opt-in."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.runtime.helpers import HelperSettings, nn_command

ROOT = Path("/repo")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _settings(env):
    return HelperSettings.from_env({"NN_TRAINER_ENABLED": "true", **env}, torch_available=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_command_has_no_arch_flags():
    cmd = nn_command("py", ROOT, "XAUUSDm", _settings({}), 1)
    assert "--arch" not in cmd and "--preset" not in cmd


def test_dual_arch_adds_flags_with_preset():
    cmd = nn_command("py", ROOT, "XAUUSDm", _settings({"NN_ARCH": "dual", "NN_PRESET": "medium"}), 1)
    assert cmd[cmd.index("--arch") + 1] == "dual"
    assert cmd[cmd.index("--preset") + 1] == "medium"


def test_loop_script_accepts_the_new_flags():
    src = (SCRIPTS / "train_nn_loop.py").read_text()
    assert '"--arch"' in src and '"--preset"' in src


def test_train_deep_cli_refuses_too_little_data(tmp_path, capsys):
    pytest.importorskip("torch")
    mod = _load("train_deep_nn")
    idx = pd.date_range("2025-03-03", periods=200, freq="5min")
    c = 2000 + np.cumsum(np.random.default_rng(0).normal(size=200))
    pd.DataFrame({"time": idx, "open": c, "high": c + 1, "low": c - 1, "close": c,
                  "tick_volume": 100}).to_csv(tmp_path / "b.csv", index=False)
    rc = mod.main(["--csv", str(tmp_path / "b.csv"), "--out", str(tmp_path / "m.pt")])
    out = capsys.readouterr().out
    assert rc == 1 and "not enough data" in out and "parameters" in out
    assert not (tmp_path / "m.pt").exists()
