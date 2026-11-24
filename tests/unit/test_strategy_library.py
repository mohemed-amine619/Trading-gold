"""
tests/unit/test_strategy_library.py — the full strategy catalogue (core/strategy/library).

What must hold for EVERY library strategy:
* no look-ahead: the signal on bar t is the same whether or not later bars exist,
  and garbage after t changes nothing up to t;
* it actually fires on realistic-shaped data (so the look-ahead test proves something);
* through the engine, with all ~118 strategies tested at once, random walks almost
  never produce a "proven" strategy (the multiple-testing bar rises with the count).

Plus: the catalogue covers every name in the brief, group tokens / aliases resolve,
and the engine stays fast enough for the live loop.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.strategy.analog import _atr
from core.strategy.library import (
    ALIASES, CATALOGUE, CATEGORIES, CLASSIC, LIBRARY_GENERATORS, category_members, resolve,
)
from core.strategy.library._common import ny_clock, pivots, swings
from core.strategy.library.events import nfp_times
from core.strategy.scalp import (
    GENERATORS, LIBRARY_STRATEGIES, ScalpConfig, ScalpEngine, required_z,
)


def market(n=2500, seed=0, start="2026-01-05 00:00", freq="15min", gap_every_week=True):
    """A random walk shaped like a 24/5 market: weekends removed (with an opening gap),
    opens that differ a little from the previous close, tick volume."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(n * 1.5), freq=freq)
    idx = idx[idx.dayofweek < 5][:n]
    steps = rng.normal(0, 1.0, n)
    new_week = np.r_[False, np.diff(idx.isocalendar().week.to_numpy().astype(int)) != 0]
    if gap_every_week:
        steps[new_week] += rng.choice([-1, 1], new_week.sum()) * rng.uniform(1.5, 4.0, new_week.sum())
    close = 2000 + np.cumsum(steps)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.15, n)
    open_[new_week] = close[new_week] - steps[new_week] * 0.1
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(50, 500, n).astype(float)}, index=idx)


def event_cfg(df):
    """News events every 2 days at 13:30 UTC, half of them titled CPI/FOMC/rate."""
    days = pd.date_range(df.index[0].normalize(), df.index[-1], freq="2D")
    titles = ["CPI m/m", "FOMC Statement", "Federal Funds Rate", "Retail Sales"]
    ev = tuple((d.to_pydatetime().replace(hour=13, minute=30, tzinfo=timezone.utc), titles[i % 4])
               for i, d in enumerate(days))
    return ScalpConfig(events=ev)


def with_news_spikes(df, cfg):
    """At every event: a release bar that spikes 6 points out of the hour-before range
    and closes back at its middle (what the reversal rules look for)."""
    df = df.copy()
    for k, (when, _) in enumerate(cfg.events):
        t = pd.Timestamp(when).tz_convert(None)
        if t not in df.index:
            continue
        i = df.index.get_loc(t)
        pre = df.iloc[i - 4:i]
        mid = (pre["high"].max() + pre["low"].min()) / 2
        df.iloc[i, df.columns.get_loc("close")] = mid
        if k % 2:
            df.iloc[i, df.columns.get_loc("high")] = pre["high"].max() + 6.0
            df.iloc[i, df.columns.get_loc("low")] = min(mid, df.iloc[i]["open"]) - 0.1
        else:
            df.iloc[i, df.columns.get_loc("low")] = pre["low"].min() - 6.0
            df.iloc[i, df.columns.get_loc("high")] = max(mid, df.iloc[i]["open"]) + 0.1
    return df


CFG = ScalpConfig()
EV_CFG = event_cfg(market())
DF = with_news_spikes(market(), EV_CFG)
EVENT_STRATEGIES = ("news_breakout", "news_reversal", "nfp", "cpi", "fomc", "rate_decision")


def sig(name, df, cfg=None):
    cfg = cfg or (EV_CFG if name in EVENT_STRATEGIES else CFG)
    return GENERATORS[name](df, _atr(df), cfg)


# ── every strategy ───────────────────────────────────────────────────────────

class TestEveryStrategy:
    @pytest.mark.parametrize("name", LIBRARY_STRATEGIES)
    def test_returns_one_int8_signal_per_bar(self, name):
        s = sig(name, DF)
        assert len(s) == len(DF) and s.dtype == np.int8 and set(np.unique(s)) <= {-1, 0, 1}

    @pytest.mark.parametrize("name", LIBRARY_STRATEGIES)
    def test_fires_on_market_shaped_data(self, name):
        fired = int((sig(name, DF) != 0).sum())
        if name in ("cup_handle", "nfp"):
            # rare patterns: check across more data
            big = market(6000, seed=11)
            fired = int((sig(name, big, event_cfg(big) if name == "nfp" else None) != 0).sum())
        assert fired > 0, f"{name} never fires; its tests would prove nothing"

    @pytest.mark.parametrize("name", LIBRARY_STRATEGIES)
    def test_a_signal_never_depends_on_later_bars(self, name):
        df = DF.iloc[:1600]
        full = sig(name, df)
        fired = [t for t in np.flatnonzero(full) if t > 250][:25]
        probes = sorted({x for f in fired for x in (f - 1, f)} | {700, 1100, 1599})
        for t in probes:
            assert sig(name, df.iloc[:t + 1])[-1] == full[t], f"{name} changed at bar {t}"

    @pytest.mark.parametrize("name", LIBRARY_STRATEGIES)
    def test_garbage_after_a_bar_does_not_change_its_signal(self, name):
        df = DF.iloc[:1400]
        rng = np.random.default_rng(5)
        t = 1000
        g = df.copy()
        noise = 2000 + rng.normal(0, 40, (len(df) - t - 1, 4))
        g.iloc[t + 1:, :4] = np.sort(noise, axis=1)[:, [1, 3, 0, 2]]     # o,h,l,c consistent
        g.iloc[t + 1:, 4] = rng.integers(1, 5000, len(df) - t - 1)
        assert sig(name, g)[:t + 1].tolist() == sig(name, df)[:t + 1].tolist()


# ── building blocks ──────────────────────────────────────────────────────────

class TestBlocks:
    def test_a_pivot_is_published_k_bars_after_it_happens(self):
        high = np.array([1, 2, 3, 9, 3, 2, 1, 2, 3], dtype=float)
        p = pivots(high, high - 1, 3)
        hi = p.kind == 1
        assert list(p.idx[hi]) == [3] and list(p.conf[hi]) == [6]
        sw = swings(high, high - 1, 3)
        assert np.isnan(sw.sh1[5]) and sw.sh1[6] == 9

    def test_ny_clock_handles_daylight_saving(self):
        winter = ny_clock(pd.DatetimeIndex(["2026-01-15 13:30"]), 0.0)
        summer = ny_clock(pd.DatetimeIndex(["2026-07-15 13:30"]), 0.0)
        assert winter.ny_hour[0] == 8.5 and summer.ny_hour[0] == 9.5

    def test_server_offset_shifts_the_clock(self):
        c = ny_clock(pd.DatetimeIndex(["2026-01-15 15:30"]), 2.0)     # UTC+2 broker
        assert c.ny_hour[0] == 8.5

    def test_the_trading_day_starts_at_18_new_york(self):
        c = ny_clock(pd.DatetimeIndex(["2026-01-15 22:45", "2026-01-15 23:00"]), 0.0)
        assert c.fx_day[0] != c.fx_day[1]                               # 17:45 vs 18:00 NY

    def test_nfp_is_the_first_friday_at_0830_new_york(self):
        times = nfp_times(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-31"))
        assert pd.Timestamp("2026-01-02 13:30") in times                # EST
        assert pd.Timestamp("2026-07-03 12:30") in times                # EDT


# ── catalogue / names ────────────────────────────────────────────────────────

BRIEF_COUNTS = {"price_action": 16, "smc": 21, "ict": 12, "indicators": 20, "scalping": 14,
                "daytrading": 12, "swing": 14, "quant": 18, "volume": 14, "events": 10}


class TestCatalogue:
    def test_every_name_in_the_brief_is_accounted_for(self):
        counts = {c: sum(e.category == c for e in CATALOGUE) for c in CATEGORIES}
        assert counts == BRIEF_COUNTS and len(CATALOGUE) == 151

    def test_every_signal_entry_points_at_a_real_generator(self):
        for e in CATALOGUE:
            if e.kind == "signal":
                assert e.target in GENERATORS, e
            elif e.kind == "unavailable":
                assert e.target is None and e.note, e

    def test_every_library_generator_is_reachable_from_the_catalogue(self):
        targets = {e.target for e in CATALOGUE if e.kind == "signal"}
        assert set(LIBRARY_STRATEGIES) <= targets

    def test_sizing_rules_are_listed(self):
        sizing = {e.target for e in CATALOGUE if e.kind == "sizing"}
        assert sizing == {"grid", "martingale", "anti_martingale", "dca", "kelly"}

    def test_group_tokens_and_aliases_resolve(self):
        assert resolve(["classic"]) == CLASSIC
        assert len(resolve(["all"])) == len(CLASSIC) + len(LIBRARY_GENERATORS)
        assert resolve(["smc"]) == category_members("smc")
        assert resolve(["fair_value_gap_fvg", "FVG", "fvg"]) == ("fvg",)
        assert resolve(["burst", "nonsense", "  "]) == ("burst",)
        assert ALIASES["ict_silver_bullet"] == "silver_bullet"


# ── through the engine ───────────────────────────────────────────────────────

class TestEngine:
    def test_the_bar_rises_with_the_number_of_strategies(self):
        assert required_z(7, 0.0) == pytest.approx(2.69, abs=0.01)
        assert required_z(118, 2.7) > 3.4
        assert required_z(1, 2.7) == 2.7

    def test_auto_z_is_applied_and_reported(self):
        eng = ScalpEngine(ScalpConfig(strategies=("all",), auto_z=True))
        assert eng.min_z == pytest.approx(required_z(len(eng.names), 2.7))
        rep = eng.evaluate(market(1200, seed=2)).report
        assert len(rep) == len(eng.names) and all(r["min_z"] == eng.min_z for r in rep.values())

    def test_without_auto_z_the_configured_bar_is_kept(self):
        eng = ScalpEngine(ScalpConfig(strategies=("all",)))
        assert eng.min_z == 2.7

    def test_random_walks_almost_never_prove_anything_with_all_strategies(self):
        eng = ScalpEngine(ScalpConfig(strategies=("all",), auto_z=True))
        hits = 0
        for s in range(12):
            rep = eng.evaluate(market(1500, seed=100 + s)).report
            hits += any(r.get("proven") for r in rep.values())
        assert hits <= 2

    def test_a_planted_edge_is_still_found_among_all_strategies(self):
        from tests.unit.test_scalp_strategies import planted_burst_series
        eng = ScalpEngine(ScalpConfig(strategies=("all",), auto_z=True))
        res = eng.evaluate(planted_burst_series(last_is_burst=True))
        assert res.report["burst"]["proven"] is True
        assert any(v.name == "burst" and v.direction == 1 for v in res.votes)

    def test_fast_enough_and_cached_between_identical_frames(self):
        df = market(3000, seed=1)
        eng = ScalpEngine(ScalpConfig(strategies=("all",), auto_z=True))
        t0 = time.perf_counter()
        first = eng.evaluate(df)
        t1 = time.perf_counter()
        second = eng.evaluate(df.copy())
        t2 = time.perf_counter()
        assert t1 - t0 < 6.0
        assert second is first and t2 - t1 < 0.05

    def test_a_new_bar_invalidates_the_cache(self):
        df = market(1500, seed=4)
        eng = ScalpEngine(ScalpConfig(strategies=("bos",)))
        a = eng.evaluate(df.iloc[:-1])
        b = eng.evaluate(df)
        assert a is not b

    def test_a_broken_strategy_abstains_instead_of_stopping_the_rest(self, monkeypatch):
        def boom(df, atr, cfg):
            raise ValueError("bad rule")
        monkeypatch.setitem(GENERATORS, "bos", boom)
        res = ScalpEngine(ScalpConfig(strategies=("bos", "choch"))).evaluate(market(1200))
        assert set(res.report) == {"bos", "choch"} and not res.report["bos"]["proven"]
