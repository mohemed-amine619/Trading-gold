"""Why the NN had no model: the trainer must judge at the bot's confidence, on the
bot's timeframe, and the bot must say why there is no model file."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ai.nn_predictor import ModelPredictor
from core.runtime.helpers import HelperSettings, nn_command


def _bars(n=500):
    c = 2000 + np.cumsum(np.random.default_rng(0).normal(size=n))
    idx = pd.date_range("2025-03-03", periods=n, freq="15min")
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                         "tick_volume": 100.0}, index=idx)


def _cmd(env):
    s = HelperSettings.from_env({"NN_TRAINER_ENABLED": "true", **env}, True)
    return nn_command("py", Path("/bot"), "XAUUSDm", s, 1)


def test_trainer_threshold_follows_nn_min_confidence():
    cmd = _cmd({"NN_MIN_CONFIDENCE": "0.6"})
    assert cmd[cmd.index("--threshold") + 1] == "0.6"


def test_holdout_bars_is_configurable():
    cmd = _cmd({"NN_TRAINER_HOLDOUT_BARS": "3000"})
    assert cmd[cmd.index("--holdout-bars") + 1] == "3000"


def test_unset_keeps_the_trainer_defaults():
    cmd = _cmd({})
    assert "--threshold" not in cmd and "--holdout-bars" not in cmd


def test_trainer_runs_on_the_timeframe_it_is_given():
    cmd = _cmd({"TIMEFRAME": "M15"})
    assert cmd[cmd.index("--timeframe") + 1] == "M15"
    assert cmd[cmd.index("--out") + 1].endswith("nn_XAUUSDm_M15.pt")


def test_no_model_reason_includes_the_trainers_last_verdict(tmp_path):
    model = tmp_path / "nn_XAUUSDm_M15.pt"
    hist = Path(str(model) + ".history.jsonl")
    hist.write_text(
        json.dumps({"action": "skipped", "reason": "not enough bars"}) + "\n"
        + json.dumps({"action": "kept",
                      "reason": "challenger edge unmeasured (too few holdout trades)"}) + "\n",
        encoding="utf-8")
    sig = ModelPredictor(str(model)).predict_signal(_bars())
    assert sig.direction == 0
    assert "no model file" in sig.reason
    assert "2 cycle(s)" in sig.reason and "too few holdout trades" in sig.reason


def test_no_model_and_no_history_says_the_trainer_has_not_finished(tmp_path):
    sig = ModelPredictor(str(tmp_path / "m.pt")).predict_signal(_bars())
    assert "no model file" in sig.reason and "not finished a cycle" in sig.reason
