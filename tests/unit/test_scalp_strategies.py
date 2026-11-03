"""
tests/unit/test_scalp_strategies.py — the scalping strategies and their proof gate.

Signal logic is checked on hand-built price patterns where the bar that must fire
is known in advance. The proof gate is checked on hand-built signal/outcome
arrays, with the thresholds worked out by hand. Two properties matter most and
are tested for every strategy: no look-ahead, and abstaining on random walks.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from core.strategy.analog import _atr
from core.strategy.scalp import (
    ALL_STRATEGIES, GENERATORS, ScalpConfig, ScalpEngine, ScalpVote, prove, scalp_support,
    score_signals, thin,
)

CFG = ScalpConfig()


def frame(close, high=None, low=None, open_=None, volume=None, start="2026-01-05 00:00", freq="5min"):
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "open": close if open_ is None else np.asarray(open_, dtype=float),
        "high": close + 0.2 if high is None else np.asarray(high, dtype=float),
        "low": close - 0.2 if low is None else np.asarray(low, dtype=float),
        "close": close,
        "tick_volume": 100.0 if volume is None else np.asarray(volume, dtype=float),
    }, index=idx)


def walk(n=1500, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, sd, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    return frame(close, high, low, open_, rng.integers(50, 500, n))


def signal(name, df, cfg=CFG):
    return GENERATORS[name](df, _atr(df), cfg)


# ── signal logic on hand-built patterns ──────────────────────────────────────

class TestOpeningRangeBreakout:
    def day(self, break_to):
        close = np.full(288, 2000.0)              # one day of 5-minute bars from 00:00
        high, low = close + 0.2, close - 0.2
        # opening range = the six bars from 07:00 (index 84..89): high 2001, low 1999
        high[84:90], low[84:90] = 2001.0, 1999.0
        close[90] = break_to                      # 07:30
        return frame(close, high, low)

    def test_a_close_above_the_range_fires_long_once_on_that_bar(self):
        sig = signal("orb", self.day(2002.0))
        assert sig[90] == 1 and np.flatnonzero(sig).tolist() == [90]

    def test_a_close_below_the_range_fires_short(self):
        sig = signal("orb", self.day(1998.0))
        assert sig[90] == -1 and np.flatnonzero(sig).tolist() == [90]

    def test_staying_inside_the_range_gives_no_signal(self):
        assert not signal("orb", self.day(2000.5)).any()

    def test_it_fires_at_most_once_per_day(self):
        df = self.day(2002.0)
        df.iloc[91:100, df.columns.get_loc("close")] = 2005.0       # keeps going up
        assert (signal("orb", df) != 0).sum() == 1

    def test_a_day_that_ends_before_the_range_is_complete_has_no_signal(self):
        short = self.day(2002.0).iloc[:88]
        assert not signal("orb", short).any()

    def test_the_open_hour_is_respected(self):
        sig = signal("orb", self.day(2002.0), ScalpConfig(open_hour=9))
        assert sig[90] == 0                       # the range now starts at 09:00 instead


class TestVwapFade:
    def test_a_stretch_far_above_vwap_fires_short_once(self):
        close = np.full(120, 2000.0)
        close[1::2] += 0.1                        # tiny noise so ATR exists
        close[80:] = 2004.0                       # jumps ~4 ATR above the day's VWAP
        sig = signal("vwap_fade", frame(close))
        assert sig[80] == -1 and (sig == -1).sum() == 1 and not (sig == 1).any()

    def test_a_stretch_far_below_vwap_fires_long(self):
        close = np.full(120, 2000.0)
        close[1::2] += 0.1
        close[80:] = 1996.0
        sig = signal("vwap_fade", frame(close))
        assert sig[80] == 1 and not (sig == -1).any()

    def test_no_signal_early_in_the_day_when_vwap_is_not_formed(self):
        close = np.full(40, 2000.0)
        close[1::2] += 0.1
        close[8:] = 2010.0
        assert not signal("vwap_fade", frame(close)).any()


class TestSqueeze:
    def test_a_breakout_after_a_squeeze_fires_long(self):
        rng = np.random.default_rng(1)
        close = np.r_[2000 + rng.normal(0, 3.0, 120),       # wide: teaches the threshold
                      2000 + rng.normal(0, 0.05, 40),       # squeeze
                      [2006.0]]                             # breakout bar 160
        sig = signal("squeeze", frame(close))
        assert sig[160] == 1

    def test_a_breakdown_after_a_squeeze_fires_short(self):
        rng = np.random.default_rng(1)
        close = np.r_[2000 + rng.normal(0, 3.0, 120), 2000 + rng.normal(0, 0.05, 40), [1994.0]]
        assert signal("squeeze", frame(close))[160] == -1


class TestRsi2:
    def test_a_sharp_dip_inside_an_uptrend_fires_long(self):
        close = 2000 + 0.3 * np.arange(130.0)
        close[120:122] -= [1.5, 3.5]              # two down bars inside the uptrend
        sig = signal("rsi2", frame(close))
        assert (sig[120:123] == 1).any() and not (sig == -1).any()

    def test_a_sharp_bounce_inside_a_downtrend_fires_short(self):
        close = 2000 - 0.3 * np.arange(130.0)
        close[120:122] += [1.5, 3.5]
        sig = signal("rsi2", frame(close))
        assert (sig[120:123] == -1).any() and not (sig == 1).any()

    def test_a_dip_against_no_trend_is_ignored(self):
        close = np.full(130, 2000.0)
        close[1::2] += 0.1
        close[120:122] -= [1.5, 3.5]
        assert not signal("rsi2", frame(close)).any()


class TestMomentumBurst:
    def base(self, burst_volume=2000.0, close_at=0.95):
        n = 80
        close = 2000 + np.where(np.arange(n) % 2 == 0, 0.2, -0.2)
        high, low = close + 0.5, close - 0.5      # range 1.0 everywhere
        volume = np.full(n, 100.0)
        high[60], low[60] = 2005.0, 2000.0        # range 5.0, five times normal
        close[60] = 2000.0 + close_at * 5.0
        volume[60] = burst_volume
        return frame(close, high, low, volume=volume)

    def test_a_wide_high_volume_bar_closing_near_its_high_fires_long(self):
        sig = signal("burst", self.base())
        assert sig[60] == 1 and np.flatnonzero(sig).tolist() == [60]

    def test_closing_near_the_low_fires_short(self):
        assert signal("burst", self.base(close_at=0.05))[60] == -1

    def test_without_the_volume_surge_there_is_no_signal(self):
        assert signal("burst", self.base(burst_volume=100.0))[60] == 0

    def test_closing_in_the_middle_is_not_a_burst(self):
        assert signal("burst", self.base(close_at=0.5))[60] == 0


class TestEmaPullback:
    def test_a_dip_to_the_ema21_in_an_uptrend_fires_long(self):
        close = 2000 + 0.5 * np.arange(120.0)
        low = close - 0.2
        low[100] = close[100] - 8.0               # pierces the EMA21, closes back up
        sig = signal("ema_pullback", frame(close, low=low))
        assert sig[100] == 1 and not (sig[60:100] != 0).any()

    def test_a_poke_to_the_ema21_in_a_downtrend_fires_short(self):
        close = 2000 - 0.5 * np.arange(120.0)
        high = close + 0.2
        high[100] = close[100] + 8.0
        sig = signal("ema_pullback", frame(close, high=high))
        assert sig[100] == -1 and not (sig[60:100] != 0).any()


class TestSweep:
    def test_a_new_low_that_closes_back_inside_fires_long(self):
        close = np.full(60, 2000.0)
        close[1::2] += 0.5
        low = close - 0.5
        low[40] = 1997.0                          # sweeps the 20-bar low (1999.5)
        close[40] = 2000.8
        open_ = close.copy()
        open_[40] = 2000.0                        # an up bar
        sig = signal("sweep", frame(close, low=low, open_=open_))
        assert sig[40] == 1

    def test_a_new_high_that_closes_back_inside_fires_short(self):
        close = np.full(60, 2000.0)
        close[1::2] += 0.5
        high = close + 0.5
        high[40] = 2003.0
        close[40] = 1999.4
        open_ = close.copy()
        open_[40] = 2000.2                        # a down bar
        assert signal("sweep", frame(close, high=high, open_=open_))[40] == -1

    def test_a_breakout_that_holds_is_not_a_sweep(self):
        close = np.full(60, 2000.0)
        close[1::2] += 0.5
        low = close - 0.5
        low[40] = 1997.0
        close[40] = 1997.5                        # closes below the old low: a real break
        assert signal("sweep", frame(close, low=low))[40] == 0


# ── no look-ahead, for every strategy ────────────────────────────────────────

class TestCausality:
    @pytest.mark.parametrize("name", ALL_STRATEGIES)
    def test_a_signal_never_depends_on_later_bars(self, name):
        df = walk(1100, seed=7)
        full = signal(name, df)
        # Check the bars where a signal actually fires (and the bar before each), not
        # arbitrary ones: a look-ahead only shows where a signal depends on it.
        fired = [t for t in np.flatnonzero(full) if t > 60]
        assert fired, f"{name} fired nothing on the test series, so this test would prove nothing"
        for t in sorted({x for f in fired[:40] for x in (f - 1, f)}):
            assert signal(name, df.iloc[:t + 1])[-1] == full[t], f"{name} changed at bar {t}"

    @pytest.mark.parametrize("name", ALL_STRATEGIES)
    def test_garbage_after_a_bar_does_not_change_its_signal(self, name):
        df = walk(900, seed=3)
        rng = np.random.default_rng(9)
        t = 600
        garbage = df.copy()
        garbage.iloc[t + 1:, :4] = 2000 + rng.normal(0, 40, (len(df) - t - 1, 4))
        assert signal(name, garbage)[:t + 1].tolist() == signal(name, df)[:t + 1].tolist()


# ── the proof gate, on hand-built arrays ─────────────────────────────────────

def arrays(n, calls, wins):
    """Signals at ``calls`` (all long); outcomes: up on the first ``wins`` calls, down on the rest."""
    sig = np.zeros(n, dtype=np.int8)
    out = np.full(n, np.nan)
    for k, t in enumerate(calls):
        sig[t] = 1
        out[t] = 1.0 if k < wins else 0.0
    return sig, out


GATE = ScalpConfig(horizon=12, check_bars=250, min_calls=15, min_edge=0.10, min_z=2.7)
# n=300, horizon 12 -> the replay window is bars 38..287, so these 20 calls are all inside it
CALLS20 = list(range(40, 280, 12))


class TestHelpers:
    def test_thin_keeps_calls_at_least_a_horizon_apart(self):
        assert thin([0, 1, 2, 12, 13, 30], 12) == [0, 12, 30]
        assert thin([], 12) == []

    def test_score_signals_counts_wins_losses_and_flat_calls(self):
        sig = np.array([0, 1, 0, -1, 1, 1, 0])
        out = np.array([np.nan, 1.0, np.nan, 0.0, 0.0, np.nan, np.nan])
        assert score_signals(sig, out, 0, 6) == (2, 1, 4)    # win, win, loss, one timeout


class TestProve:
    def test_all_wins_across_twenty_independent_calls_is_proven(self):
        sig, out = arrays(300, CALLS20, wins=20)
        r = prove(sig, out, GATE)
        assert r["proven"] and r["edge"] == 1.0 and r["n_independent"] == 20

    def test_the_boundary_is_2_7_over_root_n(self):
        calls = CALLS20                      # 20 calls: needs 2.7/sqrt(20) = 0.6037
        sig, out = arrays(300, calls, wins=17)                # edge (17-3)/20 = 0.70
        assert prove(sig, out, GATE)["proven"] is True
        sig, out = arrays(300, calls, wins=16)                # edge (16-4)/20 = 0.60
        r = prove(sig, out, GATE)
        assert r["proven"] is False and r["required"] == pytest.approx(0.6037, abs=0.001)
        assert "no proven edge" in r["reason"]

    def test_a_zero_edge_is_not_proven(self):
        sig, out = arrays(300, CALLS20, wins=10)
        assert prove(sig, out, GATE)["proven"] is False

    def test_too_few_independent_calls_is_unproven_not_assumed_good(self):
        sig, out = arrays(300, list(range(40, 160, 12)), wins=10)      # 10 calls
        r = prove(sig, out, GATE)
        assert r["proven"] is False and r["edge"] is None and "unproven" in r["reason"]

    def test_clustered_calls_do_not_count_as_independent(self):
        # 40 calls, but in four bursts of ten consecutive bars: only 4 independent events
        calls = [s + i for s in (50, 100, 150, 200) for i in range(10)]
        sig, out = arrays(300, calls, wins=40)
        r = prove(sig, out, GATE)
        assert r["calls"] == 40 and r["n_independent"] == 4 and r["proven"] is False

    def test_calls_in_the_newest_horizon_bars_are_never_scored(self):
        calls = CALLS20 + [290, 295, 299]    # the last three have no outcome yet
        sig, out = arrays(300, calls, wins=20)
        r = prove(sig, out, GATE)
        assert r["calls"] == 20

    def test_only_the_recent_window_is_replayed(self):
        # perfect calls long ago, nothing recently: the old record must not count
        sig, out = arrays(600, list(range(20, 140, 6)), wins=20)
        assert prove(sig, out, GATE)["n_independent"] == 0


# ── the engine ───────────────────────────────────────────────────────────────

def planted_burst_series(n=1200, last_is_burst=True, seed=0, back=0):
    """Noise plus a momentum burst every 30 bars that really is followed by continuation.

    Closes are built first (every shift applied), then highs and lows are derived
    from the final closes, then the burst bars get their wide range and volume.
    """
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 0.25, n))
    volume = rng.integers(80, 120, n).astype(float)
    bursts = list(range(60, n - 20, 30))
    if last_is_burst:
        bursts.append(n - 1 - back)                                 # `back` bars before the newest
    for b in bursts:
        close[b:] += 6.0                                          # the burst bar jumps up
        end = min(b + 13, n)
        if b < n - 1:
            close[b + 1:end] += 0.7 * np.arange(1, end - b)       # then keeps going for 12 bars
            close[end:] += 0.7 * (end - b - 1)
    high, low = close + 0.3, close - 0.3
    for b in bursts:
        high[b], low[b] = close[b] + 0.05, close[b] - 6.0         # wide bar closing at its high
        volume[b] = 2000.0
    return frame(close, high, low, volume=volume)


class TestEngine:
    def test_a_strategy_with_real_edge_is_proven_and_votes_on_a_live_signal(self):
        engine = ScalpEngine(ScalpConfig(strategies=("burst",)))
        res = engine.evaluate(planted_burst_series(last_is_burst=True))
        assert res.report["burst"]["proven"] is True and res.report["burst"]["edge"] > 0.5
        assert [(v.name, v.direction) for v in res.votes] == [("burst", 1)]
        assert 0.6 <= res.votes[0].strength <= 1.0

    def test_a_proven_strategy_without_a_live_signal_does_not_vote(self):
        engine = ScalpEngine(ScalpConfig(strategies=("burst",)))
        res = engine.evaluate(planted_burst_series(last_is_burst=False))
        assert res.report["burst"]["proven"] is True
        assert res.votes == []

    def test_the_signal_stays_live_for_valid_bars_only(self):
        engine = ScalpEngine(ScalpConfig(strategies=("burst",), valid_bars=2))
        assert engine.evaluate(planted_burst_series(back=0)).votes      # on the newest bar
        assert engine.evaluate(planted_burst_series(back=1)).votes      # one bar ago: still live
        assert engine.evaluate(planted_burst_series(back=2)).votes == []   # two bars ago: stale

    def test_random_walks_almost_never_prove_any_strategy(self):
        engine = ScalpEngine()
        proven = [s for s in range(100)
                  if any(v.get("proven") for v in engine.evaluate(walk(1500, s)).report.values())]
        assert len(proven) <= 2                           # measured: 2 of 300 at min_z=2.7

    def test_a_lower_bar_lets_noise_through_which_is_why_it_is_2_7(self):
        loose = ScalpEngine(ScalpConfig(min_z=2.0))
        slipped = sum(any(v.get("proven") for v in loose.evaluate(walk(1500, s)).report.values())
                      for s in range(100))
        assert slipped >= 3                               # measured: 19 of 300 at min_z=2.0

    def test_too_little_history_proves_nothing(self):
        res = ScalpEngine().evaluate(walk(40))
        assert res.votes == [] and all("not enough history" in v["reason"] for v in res.report.values())

    def test_a_frame_without_a_datetime_index_is_refused_not_crashed(self):
        res = ScalpEngine().evaluate(walk(500).reset_index(drop=True))
        assert res.votes == [] and all(not v["proven"] for v in res.report.values())

    def test_strategy_selection_and_unknown_names(self):
        res = ScalpEngine(ScalpConfig(strategies=("burst", "does_not_exist"))).evaluate(walk(800))
        assert set(res.report) == {"burst"}

    def test_it_is_fast_enough_to_run_once_per_closed_bar(self):
        df = walk(3000, 1)
        start = time.perf_counter()
        ScalpEngine().evaluate(df)
        assert time.perf_counter() - start < 2.0

    def test_vote_strength_grows_with_the_proven_edge(self):
        engine = ScalpEngine(ScalpConfig(strategies=("burst",)))
        v = engine.evaluate(planted_burst_series()).votes[0]
        assert v.strength == pytest.approx(0.6 + 0.4 * min(1.0, v.edge))


class TestSupport:
    def vote(self, direction, strength):
        return ScalpVote("x", direction, strength, 0.5, 20)

    def test_no_live_votes_means_the_component_is_absent(self):
        assert scalp_support([], 1) is None

    def test_one_agreeing_vote_gives_its_strength_and_none_to_the_opposite(self):
        votes = [self.vote(1, 0.8)]
        assert scalp_support(votes, 1) == pytest.approx(0.8)
        assert scalp_support(votes, -1) == 0.0

    def test_two_agreeing_votes_average(self):
        assert scalp_support([self.vote(1, 0.8), self.vote(1, 0.6)], 1) == pytest.approx(0.7)

    def test_a_mixed_vote_is_the_net(self):
        assert scalp_support([self.vote(1, 0.8), self.vote(-1, 0.6)], 1) == pytest.approx(0.1)
        assert scalp_support([self.vote(1, 0.8), self.vote(-1, 0.6)], -1) == 0.0

    def test_never_outside_zero_to_one(self):
        for votes in ([self.vote(1, 1.0)] * 3, [self.vote(-1, 1.0)] * 3):
            for d in (1, -1):
                assert 0.0 <= scalp_support(votes, d) <= 1.0
