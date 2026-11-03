"""
tests/unit/test_nn_integration.py — the neural-network vote inside the bot.

Safety property: with no model, no measured edge, or no confident call, the
component is absent and every score is exactly what it was without it.
"""

from __future__ import annotations

import os
import types

import pandas as pd
import pytest

from core.ai.nn_predictor import NNSignal
from core.config.schema import PARAMS
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_nn_features import random_walk
from tests.unit.test_risk_ladder import bull_frame

NEUTRAL = NNSignal(0, 0.5, (1.0, 0.0, 0.0), "no model file at x")


def vote(direction, confidence=0.8):
    probs = (0.1, confidence if direction == 1 else 0.1, confidence if direction == -1 else 0.1)
    return NNSignal(direction, confidence, probs, "signal")


class TestScore:
    def cfg(self, run_module, **over):
        cfg = dict(run_module.CONFIG)
        cfg.update(fib_enabled=False, trend_weight=30.0, momentum_weight=15.0, adx_strong=30.0,
                   analog_enabled=False, scalp_enabled=False, nn_enabled=True, nn_weight=15.0)
        cfg.update(over)
        return cfg

    def score(self, run_module, nn, direction=1, **over):
        m = bull_frame()
        return run_module._score_direction(1, 0.8, m, m, m, direction, self.cfg(run_module, **over),
                                           nn=nn)

    def test_no_signal_or_a_neutral_one_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        base = self.score(run_module, None)
        assert self.score(run_module, NEUTRAL) == base
        m = bull_frame()
        assert base == run_module._score_direction(1, 0.8, m, m, m, 1, self.cfg(run_module))

    def test_disabled_or_zero_weight_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        base = self.score(run_module, None)
        assert self.score(run_module, vote(1), nn_enabled=False) == base
        assert self.score(run_module, vote(1), nn_weight=0.0) == base

    def test_hand_computed_scores(self, run_and_stub):
        run_module, _ = run_and_stub
        base = self.score(run_module, None)               # nominal 100, factor 1: the raw sum
        k = 100.0 / 115.0
        assert self.score(run_module, vote(1, 0.8)) == pytest.approx((base + 15 * 0.8) * k)
        assert self.score(run_module, vote(-1, 0.8)) == pytest.approx(base * k)    # opposed: 0 pts

    def test_more_confidence_scores_higher_and_opposing_scores_lowest(self, run_and_stub):
        run_module, _ = run_and_stub
        high, low = self.score(run_module, vote(1, 0.95)), self.score(run_module, vote(1, 0.73))
        assert high > low > self.score(run_module, vote(-1, 0.95))

    def test_the_total_never_exceeds_one_hundred(self, run_and_stub):
        run_module, _ = run_and_stub
        m = bull_frame()
        for nn in (None, NEUTRAL, vote(1, 1.0), vote(-1, 1.0)):
            for conf in (0.0, 0.5, 1.0):
                got = run_module._score_direction(1, conf, m, m, m, 1, self.cfg(run_module), nn=nn)
                assert 0.0 <= got <= 100.0 + 1e-9

    def test_it_combines_with_the_other_components(self, run_and_stub):
        run_module, _ = run_and_stub
        from core.strategy.scalp import ScalpVote
        m = bull_frame()
        cfg = self.cfg(run_module, scalp_enabled=True, scalp_weight=15.0)
        base = run_module._score_direction(1, 0.8, m, m, m, 1, cfg)
        both = run_module._score_direction(1, 0.8, m, m, m, 1, cfg, nn=vote(1, 0.8),
                                           scalp=[ScalpVote("burst", 1, 0.8, 0.7, 20)])
        assert both == pytest.approx((base + 15 * 0.8 + 15 * 0.8) * 100.0 / 130.0)


class TestBot:
    def bot(self, run_module, **over):
        cfg = dict(run_module.CONFIG)
        cfg.update(symbol="XAUUSDm", nn_enabled=True)
        cfg.update(over)
        return run_module.StandaloneAIBot(cfg)

    def test_keys_are_declared_with_safe_defaults(self, run_and_stub):
        run_module, _ = run_and_stub
        for key in ("nn_enabled", "nn_weight", "nn_min_confidence", "nn_min_oos_edge"):
            assert key in run_module.CONFIG and key in PARAMS, key
        assert run_module.CONFIG["nn_min_confidence"] == 0.72
        assert run_module.CONFIG["nn_min_oos_edge"] == 0.05      # stricter than the trainer's floor

    def test_the_model_path_matches_where_the_trainer_saves(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        monkeypatch.setenv("TIMEFRAME", "m15")
        path = self.bot(run_module)._nn_model_path()
        assert path.endswith(os.path.join("data", "models", "nn_XAUUSDm_M15.pt"))
        from core.runtime.helpers import HelperSettings, nn_command
        from pathlib import Path
        cmd = nn_command("py", Path(path).parents[2], "XAUUSDm", HelperSettings.from_env(
            {"NN_TRAINER_ENABLED": "true", "TIMEFRAME": "M15"}, True), 1)
        assert cmd[cmd.index("--out") + 1] == path

    def test_disabled_returns_none(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.bot(run_module, nn_enabled=False)._get_nn(random_walk(600)) is None

    def test_no_model_file_is_neutral_and_says_so_once(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        monkeypatch.setattr(bot, "_nn_model_path", lambda: "/nonexistent/nn_XAUUSDm_M5.pt")
        with caplog.at_level("INFO"):
            for _ in range(5):
                sig = bot._get_nn(random_walk(600))
        assert sig.direction == 0
        said = [r.message for r in caplog.records if "NN not voting" in r.message]
        assert len(said) == 1 and "no model file" in said[0]

    def test_a_failure_is_swallowed_and_warned_once(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        monkeypatch.setattr(bot, "_nn_model_path", lambda: "/nonexistent/x.pt")
        bot._get_nn(random_walk(600))
        bot._nn_predictor.predict_signal = lambda df: (_ for _ in ()).throw(RuntimeError("boom"))
        with caplog.at_level("WARNING"):
            assert bot._get_nn(random_walk(600)) is None
            assert bot._get_nn(random_walk(600)) is None
        assert sum("NN vote failed" in r.message for r in caplog.records) == 1

    def test_changing_a_setting_rebuilds_the_predictor(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        monkeypatch.setattr(bot, "_nn_model_path", lambda: "/nonexistent/x.pt")
        bot._get_nn(random_walk(300))
        first = bot._nn_predictor
        bot.cfg["nn_min_confidence"] = 0.8
        bot._get_nn(random_walk(300))
        assert bot._nn_predictor is not first and bot._nn_predictor.min_confidence == 0.8

    def signal_setup(self, run_module, bot, monkeypatch):
        monkeypatch.setattr(bot, "_location_for", lambda df: None)
        monkeypatch.setattr(bot, "_get_analog", lambda df: None)
        monkeypatch.setattr(bot, "_get_scalp", lambda df: None)
        bot.ai = types.SimpleNamespace(update=lambda df: None,
                                       predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.65))
        monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
        return pd.DataFrame({
            "open": [2630.0] * 60, "high": [2632.0] * 60, "low": [2628.0] * 60, "close": [2630.0] * 60,
            "adx": [25.0] * 60, "atr": [2.0] * 60, "ema_fast": [2631.0] * 60, "ema_slow": [2629.0] * 60,
            "volume": [1000] * 60, "rsi": [55.0] * 60})

    def test_the_signal_path_hands_the_vote_to_the_scorer_and_logs_it(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, min_signal_score=40, ai_confidence_threshold=0.50)
        df = self.signal_setup(run_module, bot, monkeypatch)
        monkeypatch.setattr(bot, "_get_nn", lambda d: vote(-1, 0.81))
        bot._nn_predictor = types.SimpleNamespace(oos_edge=0.12)
        seen = []
        monkeypatch.setattr(run_module, "_score_direction",
                            lambda *a, **k: (seen.append(k.get("nn")) or 65.0))
        with caplog.at_level("INFO"):
            bot._get_signal(df, df, df)
        assert seen and all(s.direction == -1 for s in seen)
        assert any("NN vote: SELL (confidence 0.81, out-of-sample edge +0.12)" in r.message
                   for r in caplog.records)


torch = pytest.importorskip("torch")
# torch imports torch._dynamo lazily on the first optimizer, and that import scans
# sys.modules. The MT5 stub the bot tests install (a module whose __file__ is 0)
# breaks that scan, which a real process never sees. Warm it up at collection time,
# before any stub exists.
torch.optim.Adam([torch.zeros(1, requires_grad=True)])


@pytest.fixture(scope="module")
def trained():
    from core.ai.nn_model import TrainConfig, fit_artifact
    cfg = TrainConfig(window=8, hidden=8, epochs=2, batch=64, horizon=6, patience=2, n_models=2)
    return fit_artifact(random_walk(n=1500, seed=3), cfg)


def save_model(directory, artifact, edge):
    from core.ai.nn_model import save_artifact
    art = dict(artifact)
    art["oos"] = {"edge": edge}
    path = directory / "nn_XAUUSDm_M5.pt"
    save_artifact(str(path), art)
    return str(path)


def model_bot(run_module, path):
    cfg = dict(run_module.CONFIG)
    cfg.update(symbol="XAUUSDm", nn_enabled=True, nn_min_confidence=0.0, nn_min_oos_edge=0.05)
    b = run_module.StandaloneAIBot(cfg)
    b._nn_model_path = lambda: path
    return b


class TestWithARealModel:
    def test_a_model_with_a_measured_edge_gives_a_real_signal(self, run_and_stub, trained, tmp_path):
        run_module, _ = run_and_stub
        path = save_model(tmp_path, trained, 0.20)
        sig = model_bot(run_module, path)._get_nn(random_walk(n=1500, seed=3))
        assert sig.direction in (-1, 1) or sig.reason == "model favours no trade"
        assert sum(sig.probs) == pytest.approx(1.0, abs=1e-6)

    def test_an_edge_below_the_bots_floor_does_not_vote(self, run_and_stub, trained, tmp_path):
        run_module, _ = run_and_stub
        path = save_model(tmp_path, trained, 0.01)           # positive, but below nn_min_oos_edge
        sig = model_bot(run_module, path)._get_nn(random_walk(n=1500, seed=3))
        assert sig.direction == 0 and "below floor" in sig.reason

    def test_an_unmeasured_model_does_not_vote(self, run_and_stub, trained, tmp_path):
        run_module, _ = run_and_stub
        path = save_model(tmp_path, trained, None)
        sig = model_bot(run_module, path)._get_nn(random_walk(n=1500, seed=3))
        assert sig.direction == 0 and "never measured" in sig.reason

    def test_a_replaced_model_is_picked_up_without_a_restart(self, run_and_stub, trained, tmp_path):
        run_module, _ = run_and_stub
        path = save_model(tmp_path, trained, -0.5)
        bot = model_bot(run_module, path)
        assert bot._get_nn(random_walk(n=1500, seed=3)).direction == 0
        save_model(tmp_path, trained, 0.30)                  # the trainer promotes a better model
        os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
        sig = bot._get_nn(random_walk(n=1500, seed=3))
        assert "below floor" not in sig.reason and "never measured" not in sig.reason
