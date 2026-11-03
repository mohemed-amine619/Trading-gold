"""
tests/unit/test_analog_strategy.py — the "history repeats" strategy.

Three things are proven here, because they are what make a strategy like this
safe to let near real orders:

* it finds structure that genuinely repeats (a sine wave, with the expected
  direction derived from the generating function, not from the engine);
* it ABSTAINS on a random walk, where history does not repeat;
* it never reads past the bar it is predicting, and when it abstains the bot's
  score is exactly what it was without it.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from core.strategy.analog import AnalogConfig, AnalogEngine, AnalogSignal
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import bull_frame


def frame(close):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2026-01-05", periods=len(close), freq="5min")
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
                        index=idx)


def sine(n=900, period=60, amp=10.0, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return frame(2000 + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n))


def walk(n=900, seed=0):
    rng = np.random.default_rng(seed)
    return frame(2000 + np.cumsum(rng.normal(0, 1.0, n)))


ENGINE = AnalogEngine()


class TestFindsStructureThatRepeats:
    def test_a_sine_wave_is_proven_and_votes(self):
        sig = ENGINE.evaluate(sine(900))
        assert sig.abstained is False and sig.direction != 0
        assert sig.edge is not None and sig.edge > 0.8 and sig.calls >= 30

    @pytest.mark.parametrize("n", [878, 893, 880, 905])
    def test_the_direction_matches_the_waves_own_slope(self, n):
        # Oracle from the generating function: price = sin(2*pi*t/60) rises while
        # cos > 0. Pick bars well away from the turning points, where the
        # next move is unambiguous.
        phase = (n - 1) % 60
        slope = np.cos(2 * np.pi * phase / 60)
        if abs(slope) < 0.5:
            pytest.skip("too close to a turning point for an unambiguous expectation")
        sig = ENGINE.evaluate(sine(n))
        assert sig.direction == (1 if slope > 0 else -1)
        assert sig.strength >= 0.8 and sig.n_neighbours == 15

    def test_a_repeating_series_with_more_noise_is_still_found(self):
        sig = ENGINE.evaluate(sine(900, noise=0.4, seed=3))
        assert sig.abstained is False and sig.edge > 0.5

    def test_evaluation_is_deterministic_and_fast(self):
        import time
        df = sine(900)
        start = time.perf_counter()
        a = ENGINE.evaluate(df)
        elapsed = time.perf_counter() - start
        assert a == ENGINE.evaluate(df)
        assert elapsed < 1.0                         # it runs once per closed bar


class TestAbstainsWhenHistoryDoesNotRepeat:
    def test_a_random_walk_never_proves_itself(self):
        # 300 seeds were measured while building this: 0 false proofs. 60 here.
        proven = [seed for seed in range(60) if not ENGINE.evaluate(walk(900, seed)).abstained]
        assert proven == []

    def test_the_reason_says_why(self):
        sig = ENGINE.evaluate(walk(900, 1))
        assert sig.abstained and ("no proven edge" in sig.reason or "unproven" in sig.reason)

    def test_without_the_significance_requirement_noise_does_slip_through(self):
        # The reason min_z exists: with it off, some random walks "prove" themselves.
        loose = AnalogEngine(AnalogConfig(min_z=0.0, min_edge=0.10))
        slipped = sum(not loose.evaluate(walk(900, s)).abstained for s in range(40))
        assert slipped >= 1
        assert all(ENGINE.evaluate(walk(900, s)).abstained for s in range(40))

    def test_too_little_history_abstains(self):
        sig = ENGINE.evaluate(sine(200))
        assert sig.abstained and "not enough history" in sig.reason

    def test_a_short_proof_window_is_unproven_not_assumed_good(self):
        short = AnalogEngine(AnalogConfig(check_bars=25))      # fewer replays than min_calls
        sig = short.evaluate(sine(900))
        assert sig.abstained and sig.edge is None and "unproven" in sig.reason

    def test_an_impossible_significance_bar_abstains_even_on_a_sine(self):
        strict = AnalogEngine(AnalogConfig(min_z=50.0))
        sig = strict.evaluate(sine(900))
        assert sig.abstained and "no proven edge" in sig.reason and sig.edge > 0.5

    def test_unclear_analogues_make_no_call_but_do_not_abstain(self):
        picky = AnalogEngine(AnalogConfig(max_distance_ratio=0.0))
        vectors, outcomes, _ = picky._prepare(sine(900))
        sig = picky._vote(vectors, outcomes, 899)
        assert sig.direction == 0 and sig.abstained is False and "no close analogues" in sig.reason


class TestNoLookAhead:
    def test_a_vote_at_bar_t_ignores_everything_after_t(self):
        df = sine(900)
        rng = np.random.default_rng(5)
        for t in (500, 650, 800):
            garbage = df.copy()                       # a fresh copy: only bars after t change
            garbage.iloc[t + 1:, :] = 2000 + rng.normal(0, 50, (len(df) - t - 1, 4))
            assert garbage.iloc[:t + 1].equals(df.iloc[:t + 1])
            assert not garbage.iloc[t + 1:].equals(df.iloc[t + 1:])
            assert ENGINE.predict_at(garbage, t) == ENGINE.predict_at(df, t)

    def test_the_replay_inside_the_self_check_matches_a_truncated_history(self):
        # measure_edge precomputes vectors and outcomes on the FULL frame. That is only
        # safe if the vote at bar t equals the vote computed from bars up to t alone.
        df = walk(900, 11)
        vectors, outcomes, _ = ENGINE._prepare(df)
        for t in (420, 555, 700, 840):
            assert ENGINE._vote(vectors, outcomes, t) == ENGINE.predict_at(df, t), t

    def test_the_newest_horizon_bars_are_never_in_the_library(self):
        cfg = ENGINE.cfg
        df = sine(900)
        vectors, _, _ = ENGINE._prepare(df)
        for t in (450, 700, 899):
            idx = ENGINE.library_indices(vectors, t)
            assert idx.max() <= t - cfg.horizon          # outcome fully known by t
            assert idx.min() >= cfg.window - 1
            assert t not in idx and (t - 1) not in idx

    def test_the_library_needs_enough_history(self):
        vectors, _, _ = ENGINE._prepare(sine(900))
        assert ENGINE.library_indices(vectors, 200) is None


class TestAnalogSignalContract:
    def test_strength_is_the_share_of_neighbours_that_agree(self):
        sig = ENGINE.evaluate(sine(900))
        assert sig.strength == (sig.p_long if sig.direction == 1 else sig.p_short)
        assert ENGINE.cfg.vote <= sig.strength <= 1.0

    def test_an_abstained_signal_carries_no_direction(self):
        sig = ENGINE.evaluate(walk(900, 2))
        assert sig.abstained and sig.direction == 0


# ── scoring ──────────────────────────────────────────────────────────────────

def voting(direction, strength=0.9):
    return AnalogSignal(direction, strength, strength if direction == 1 else 0.0,
                        strength if direction == -1 else 0.0, 15, False, "test", 0.9, 200)


ABSTAINED = AnalogSignal(0, 0.5, 0.0, 0.0, 0, True, "unproven", None, 0)


class TestScoreBudget:
    def cfg(self, run_module, **over):
        cfg = dict(run_module.CONFIG)
        cfg.update(fib_enabled=False, trend_weight=30.0, momentum_weight=15.0, adx_strong=30.0,
                   analog_enabled=True, analog_weight=12.0)
        cfg.update(over)
        return cfg

    def score(self, run_module, analog, direction=1, **over):
        m = bull_frame()
        return run_module._score_direction(1, 0.8, m, m, m, direction, self.cfg(run_module, **over),
                                           location=None, analog=analog)

    def test_an_abstaining_strategy_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.score(run_module, ABSTAINED) == self.score(run_module, None)

    def test_a_disabled_strategy_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        assert (self.score(run_module, voting(1), analog_enabled=False)
                == self.score(run_module, None))

    def test_zero_weight_changes_nothing_exactly(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.score(run_module, voting(1), analog_weight=0.0) == self.score(run_module, None)

    def test_calling_the_scorer_without_the_new_argument_still_works(self, run_and_stub):
        run_module, _ = run_and_stub
        m = bull_frame()
        assert run_module._score_direction(1, 0.8, m, m, m, 1, self.cfg(run_module)) \
            == self.score(run_module, None)

    def test_hand_computed_scores_for_agree_neutral_and_disagree(self, run_and_stub):
        run_module, _ = run_and_stub
        base = self.score(run_module, None)             # nominal 100, factor 1: the raw sum
        k = 100.0 / 112.0                               # 100 + analog weight 12
        assert self.score(run_module, voting(1, 0.9)) == pytest.approx((base + 12 * 0.9) * k)
        assert self.score(run_module, voting(0, 0.5)) == pytest.approx((base + 0.3 * 12) * k)
        assert self.score(run_module, voting(-1, 0.9)) == pytest.approx(base * k)

    def test_agreeing_beats_neutral_beats_disagreeing(self, run_and_stub):
        run_module, _ = run_and_stub
        agree, neutral, disagree = (self.score(run_module, voting(d)) for d in (1, 0, -1))
        assert agree > neutral > disagree

    def test_the_total_never_exceeds_one_hundred(self, run_and_stub):
        run_module, _ = run_and_stub
        m = bull_frame()
        for analog in (None, ABSTAINED, voting(1, 1.0), voting(-1, 1.0), voting(0, 0.5)):
            for conf in (0.0, 0.5, 1.0):
                got = run_module._score_direction(1, conf, m, m, m, 1, self.cfg(run_module),
                                                  analog=analog)
                assert 0.0 <= got <= 100.0 + 1e-9

    def test_the_other_direction_is_scored_against_the_vote(self, run_and_stub):
        run_module, _ = run_and_stub
        long_score = self.score(run_module, voting(1), direction=1)
        short_score = self.score(run_module, voting(1), direction=-1)
        assert long_score > short_score


class TestBotIntegration:
    def bot(self, run_module, **over):
        cfg = dict(run_module.CONFIG)
        cfg.update(symbol="XAUUSDm", analog_enabled=True)
        cfg.update(over)
        return run_module.StandaloneAIBot(cfg)

    def test_defaults_are_declared_and_on(self, run_and_stub):
        run_module, _ = run_and_stub
        from core.config.schema import PARAMS
        for key in ("analog_enabled", "analog_weight", "analog_window", "analog_k",
                    "analog_horizon", "analog_vote", "analog_min_edge", "analog_min_z",
                    "analog_check_bars", "analog_min_calls"):
            assert key in run_module.CONFIG and key in PARAMS, key
        assert run_module.CONFIG["analog_enabled"] is True
        assert PARAMS["analog_min_z"].default == 2.0

    def test_disabled_returns_none(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.bot(run_module, analog_enabled=False)._get_analog(sine(900)) is None

    def test_enabled_returns_a_signal_for_a_repeating_series(self, run_and_stub):
        run_module, _ = run_and_stub
        sig = self.bot(run_module)._get_analog(sine(900))
        assert sig is not None and sig.abstained is False and sig.direction != 0

    def test_it_abstains_on_noise_inside_the_bot(self, run_and_stub):
        run_module, _ = run_and_stub
        assert self.bot(run_module)._get_analog(walk(900, 4)).abstained is True

    def test_changing_a_setting_rebuilds_the_engine(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        bot._get_analog(sine(400))
        first = bot._analog_engine
        bot.cfg["analog_k"] = 9
        bot._get_analog(sine(400))
        assert bot._analog_engine is not first and bot._analog_engine.cfg.k == 9

    def test_a_failure_is_swallowed_and_warned_once(self, run_and_stub, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        bot._get_analog(sine(400))
        bot._analog_engine.evaluate = lambda df: (_ for _ in ()).throw(RuntimeError("boom"))
        with caplog.at_level("WARNING"):
            assert bot._get_analog(sine(400)) is None
            assert bot._get_analog(sine(400)) is None
        assert sum("History-repeats strategy failed" in r.message for r in caplog.records) == 1

    def test_the_signal_path_hands_the_vote_to_the_scorer(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot = self.bot(run_module, min_signal_score=40, ai_confidence_threshold=0.50)
        vote = voting(1)
        monkeypatch.setattr(bot, "_get_analog", lambda df: vote)
        monkeypatch.setattr(bot, "_location_for", lambda df: None)
        bot.ai = types.SimpleNamespace(update=lambda df: None,
                                       predict=lambda df: types.SimpleNamespace(direction=1, confidence=0.65))
        monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("analog"))
            return 65.0

        monkeypatch.setattr(run_module, "_score_direction", spy)
        df = pd.DataFrame({
            "open": [2630.0] * 60, "high": [2632.0] * 60, "low": [2628.0] * 60, "close": [2630.0] * 60,
            "adx": [25.0] * 60, "atr": [2.0] * 60, "ema_fast": [2631.0] * 60, "ema_slow": [2629.0] * 60,
            "volume": [1000] * 60, "rsi": [55.0] * 60})
        bot._get_signal(df, df, df)
        assert seen and all(a is vote for a in seen)

    def test_a_voting_strategy_is_logged(self, run_and_stub, monkeypatch, caplog):
        run_module, _ = run_and_stub
        bot = self.bot(run_module)
        monkeypatch.setattr(bot, "_get_analog", lambda df: voting(-1, 0.8))
        monkeypatch.setattr(bot, "_location_for", lambda df: None)
        bot.ai = types.SimpleNamespace(update=lambda df: None,
                                       predict=lambda df: types.SimpleNamespace(direction=-1, confidence=0.6))
        monkeypatch.setattr(run_module, "_session_ok", lambda cfg: True)
        monkeypatch.setattr(run_module, "_score_direction", lambda *a, **k: 10.0)
        df = pd.DataFrame({
            "open": [2630.0] * 60, "high": [2632.0] * 60, "low": [2628.0] * 60, "close": [2630.0] * 60,
            "adx": [25.0] * 60, "atr": [2.0] * 60, "ema_fast": [2631.0] * 60, "ema_slow": [2629.0] * 60,
            "volume": [1000] * 60, "rsi": [55.0] * 60})
        with caplog.at_level("INFO"):
            bot._get_signal(df, df, df)
        assert any("History repeats: SELL (80%" in r.message for r in caplog.records)
