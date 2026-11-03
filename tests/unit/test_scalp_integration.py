"""
tests/unit/test_scalp_integration.py — the scalping strategies inside the bot's
score and signal path.

The property that makes it safe: when no PROVEN strategy has a live signal, the
component is absent and every score is exactly what it was without it.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from core.config.schema import PARAMS, validate_payload
from core.strategy.scalp import ScalpVote
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import bull_frame
from tests.unit.test_scalp_strategies import planted_burst_series, walk

KEYS = ("scalp_enabled", "scalp_weight", "scalp_strategies", "scalp_horizon", "scalp_check_bars",
        "scalp_min_calls", "scalp_min_edge", "scalp_min_z", "scalp_valid_bars", "scalp_open_hour")


def vote(direction, strength=0.8, name="burst"):
    return ScalpVote(name, direction, strength, 0.7, 20)


class TestScore:
    def cfg(self, run_module, **over):
        cfg = dict(run_module.CONFIG)
        cfg.update(fib_enabled=False, trend_weight=30.0, momentum_weight=15.0, adx_strong=30.0,
                   analog_enabled=False, scalp_enabled=True, scalp_weight=15.0)
        cfg.update(over)
        return cfg

    def score(self, run_module, scalp, direction=1, **over):
        m = bull_frame()
        return run_module._score_direction(1, 0.8, m, m, m, direction, self.cfg(run_module, **over),
                                           location=None, scalp=scalp)

    def test_no_votes_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.score(run_module, None) == self.score(run_module, [])
        m = bull_frame()
        assert self.score(run_module, []) == run_module._score_direction(
            1, 0.8, m, m, m, 1, self.cfg(run_module))

    def test_a_disabled_component_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        assert (self.score(run_module, [vote(1)], scalp_enabled=False)
                == self.score(run_module, None))

    def test_zero_weight_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.score(run_module, [vote(1)], scalp_weight=0.0) == self.score(run_module, None)

    def test_hand_computed_scores(self, run_and_stub):
        run_module, _ = run_and_stub
        base = self.score(run_module, None)               # nominal 100, factor 1: the raw sum
        k = 100.0 / 115.0                                 # 100 + scalp weight 15
        assert self.score(run_module, [vote(1, 0.8)]) == pytest.approx((base + 15 * 0.8) * k)
        assert self.score(run_module, [vote(-1, 0.8)]) == pytest.approx(base * k)   # opposed: 0 pts
        # one for, one against, equal strength: net 0 -> 0 points
        assert self.score(run_module, [vote(1, 0.8), vote(-1, 0.8)]) == pytest.approx(base * k)
        # two agreeing votes average their strengths
        assert self.score(run_module, [vote(1, 0.8), vote(1, 0.6)]) == pytest.approx((base + 15 * 0.7) * k)

    def test_a_live_proven_signal_against_a_direction_lowers_that_directions_score(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.score(run_module, [vote(-1)]) < self.score(run_module, None)

    def test_agreeing_beats_none_beats_opposed(self, run_and_stub):
        run_module, _ = run_and_stub
        agree = self.score(run_module, [vote(1)])
        none = self.score(run_module, None)
        opposed = self.score(run_module, [vote(-1)])
        assert agree > opposed and none > opposed

    def test_the_total_never_exceeds_one_hundred(self, run_and_stub):
        run_module, _ = run_and_stub
        m = bull_frame()
        for votes in (None, [vote(1, 1.0)], [vote(-1, 1.0)], [vote(1, 1.0)] * 5):
            for conf in (0.0, 0.5, 1.0):
                got = run_module._score_direction(1, conf, m, m, m, 1, self.cfg(run_module), scalp=votes)
                assert 0.0 <= got <= 100.0 + 1e-9

    def test_it_combines_with_the_history_repeats_component(self, run_and_stub):
        run_module, _ = run_and_stub
        from core.strategy.analog import AnalogSignal
        analog = AnalogSignal(1, 0.9, 0.9, 0.0, 15, False, "test", 0.9, 200)
        m = bull_frame()
        cfg = self.cfg(run_module, analog_enabled=True, analog_weight=12.0)
        base = run_module._score_direction(1, 0.8, m, m, m, 1, cfg)
        both = run_module._score_direction(1, 0.8, m, m, m, 1, cfg, analog=analog, scalp=[vote(1, 0.8)])
        assert both == pytest.approx((base + 12 * 0.9 + 15 * 0.8) * 100.0 / 127.0)


class TestBot:
    def bot(self, run_module, **over):
        cfg = dict(run_module.CONFIG)
        cfg.update(symbol="XAUUSDm", scalp_enabled=True, scalp_strategies="burst")
        cfg.update(over)
        return run_module.StandaloneAIBot(cfg)

    def test_every_key_is_declared_with_a_sane_default(self, run_and_stub):
        run_module, _ = run_and_stub
        for key in KEYS:
            assert key in run_module.CONFIG and key in PARAMS, key
        assert run_module.CONFIG["scalp_enabled"] is True
        assert run_module.CONFIG["scalp_min_z"] == 2.7 and PARAMS["scalp_min_z"].default == 2.7
        assert run_module.CONFIG["scalp_weight"] == 15.0

    def test_the_strategy_list_is_a_valid_dashboard_setting(self, run_and_stub):
        values, errors = validate_payload({"scalp_strategies": "burst,sweep", "scalp_weight": 10})
        assert errors == [] and values["scalp_strategies"] == "burst,sweep"

    def test_disabled_returns_none(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.bot(run_module, scalp_enabled=False)._get_scalp(planted_burst_series()) is None

    def test_a_proven_live_strategy_returns_its_vote(self, run_and_stub):
        run_module, _ = run_and_stub
        votes = self.bot(run_module)._get_scalp(planted_burst_series())
        assert [(v.name, v.direction) for v in votes] == [("burst", 1)]

    def test_noise_returns_no_votes_but_not_none(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.bot(run_module)._get_scalp(walk(1200, 5)) == []

    def test_the_strategy_list_tolerates_spaces_and_unknown_names(self, run_and_stub):
        run_module, _ = run_and_stub
        votes = self.bot(run_module, scalp_strategies=" burst , nonsense ")._get_scalp(planted_burst_series())
        assert [v.name for v in votes] == ["burst"]

    def test_changing_a_setting_rebuilds_the_engine(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        bot._get_scalp(walk(400))
        first = bot._scalp_engine
        bot.cfg["scalp_min_z"] = 3.1
        bot._get_scalp(walk(400))
        assert bot._scalp_engine is not first and bot._scalp_engine.cfg.min_z == 3.1

    def test_a_failure_is_swallowed_and_warned_once(self, run_and_stub, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        bot._get_scalp(walk(400))
        bot._scalp_engine.evaluate = lambda df: (_ for _ in ()).throw(RuntimeError("boom"))
        with caplog.at_level("WARNING"):
            assert bot._get_scalp(walk(400)) is None
            assert bot._get_scalp(walk(400)) is None
        assert sum("Scalping strategies failed" in r.message for r in caplog.records) == 1

    def test_the_proven_set_is_logged_when_it_changes_not_every_bar(self, run_and_stub, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        with caplog.at_level("INFO"):
            bot._get_scalp(planted_burst_series())
            bot._get_scalp(planted_burst_series())
            bot._get_scalp(walk(1200, 5))
        said = [r.message for r in caplog.records if "proven on recent bars" in r.message]
        assert len(said) == 2 and "burst" in said[0] and said[1].endswith("none")

    def signal_setup(self, run_module, bot, monkeypatch):
        monkeypatch.setattr(bot, "_location_for", lambda df: None)
        monkeypatch.setattr(bot, "_get_analog", lambda df: None)
        bot.ai = types.SimpleNamespace(update=lambda df: None,
                                       predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.65))
        monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
        return pd.DataFrame({
            "open": [2630.0] * 60, "high": [2632.0] * 60, "low": [2628.0] * 60, "close": [2630.0] * 60,
            "adx": [25.0] * 60, "atr": [2.0] * 60, "ema_fast": [2631.0] * 60, "ema_slow": [2629.0] * 60,
            "volume": [1000] * 60, "rsi": [55.0] * 60})

    def test_the_signal_path_hands_the_votes_to_the_scorer(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, min_signal_score=40, ai_confidence_threshold=0.50)
        df = self.signal_setup(run_module, bot, monkeypatch)
        votes = [vote(1)]
        monkeypatch.setattr(bot, "_get_scalp", lambda d: votes)
        seen = []
        monkeypatch.setattr(run_module, "_score_direction",
                            lambda *a, **k: (seen.append(k.get("scalp")) or 65.0))
        bot._get_signal(df, df, df)
        assert seen and all(s is votes for s in seen)

    def test_live_signals_are_logged(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        df = self.signal_setup(run_module, bot, monkeypatch)
        monkeypatch.setattr(bot, "_get_scalp", lambda d: [vote(-1, 0.9, "sweep")])
        monkeypatch.setattr(run_module, "_score_direction", lambda *a, **k: 10.0)
        with caplog.at_level("INFO"):
            bot._get_signal(df, df, df)
        assert any("Scalping signals: sweep SELL (edge +0.70 over 20 independent calls)" in r.message
                   for r in caplog.records)
