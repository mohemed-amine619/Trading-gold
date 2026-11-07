"""Nothing the bot writes may grow forever (the Windows box hit 'No space left on
device'), and the NN's missing-model reason must follow the trainer."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from core.ai.nn_predictor import ModelPredictor
from core.runtime.helpers import PeriodicJob, rotate_if_large

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _diag():
    spec = importlib.util.spec_from_file_location("collect_diagnostics",
                                                  SCRIPTS / "collect_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diagnostics_keeps_only_the_newest_bundles(tmp_path):
    mod = _diag()
    for i in range(15):
        (tmp_path / f"diagnostics_20261101_{i:06d}.zip").write_bytes(b"x")
    (tmp_path / "latest_summary.md").write_text("keep me")
    removed = mod.prune_bundles(tmp_path, 10)
    left = sorted(p.name for p in tmp_path.glob("diagnostics_*.zip"))
    assert len(removed) == 5 and len(left) == 10
    assert left[0] == "diagnostics_20261101_000005.zip"           # oldest five went
    assert (tmp_path / "latest_summary.md").exists()


def test_diagnostics_keep_zero_keeps_everything(tmp_path):
    mod = _diag()
    for i in range(3):
        (tmp_path / f"diagnostics_2026110{i}_000000.zip").write_bytes(b"x")
    assert mod.prune_bundles(tmp_path, 0) == []
    assert len(list(tmp_path.glob("*.zip"))) == 3


def test_helper_log_rotates_once_too_large(tmp_path):
    log = tmp_path / "nn_trainer_XAUUSDm.log"
    log.write_bytes(b"a" * 200)
    assert rotate_if_large(log, max_bytes=100) is True
    assert not log.exists()
    assert (tmp_path / "nn_trainer_XAUUSDm.log.1").stat().st_size == 200
    log.write_bytes(b"b" * 50)
    assert rotate_if_large(log, max_bytes=100) is False and log.exists()


def test_main_log_handler_rotates(run_and_stub):
    import logging
    from logging.handlers import RotatingFileHandler
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    if handlers:                                   # basicConfig is a no-op if pytest set handlers
        assert handlers[0].maxBytes > 0 and handlers[0].backupCount >= 1
    src = (Path(__file__).resolve().parents[2] / "run.py").read_text(encoding="utf-8")
    assert 'logging.FileHandler("trading_bot.log"' not in src
    assert "_RotatingFileHandler(\"trading_bot.log\"" in src


def test_ctrl_c_during_final_diagnostics_does_not_raise():
    class _Proc:
        def wait(self, timeout=None):
            raise KeyboardInterrupt

        def poll(self):
            return 0

    job = PeriodicJob("diagnostics", ["x"], ".", interval_s=60, first_delay_s=60,
                      spawn=lambda *a, **k: _Proc())
    assert job.run_blocking(timeout_s=1) is None


def test_missing_model_reason_follows_the_trainer(tmp_path):
    model = tmp_path / "nn_XAUUSDm_M15.pt"
    idx = pd.date_range("2025-03-03", periods=500, freq="15min")
    c = 2000 + np.cumsum(np.random.default_rng(0).normal(size=500))
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                       "tick_volume": 100.0}, index=idx)
    p = ModelPredictor(str(model))
    assert "not finished a cycle" in p.predict_signal(df).reason
    Path(str(model) + ".history.jsonl").write_text(
        json.dumps({"action": "kept", "reason": "challenger edge -0.020 is not above the floor"})
        + "\n", encoding="utf-8")
    reason = p.predict_signal(df).reason                  # same predictor, later call
    assert "1 cycle(s)" in reason and "not above the floor" in reason


from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: E402,F401
