"""
tests/unit/test_strategy_panel.py — every library strategy read on one timeframe
(core/strategy/panel.py). The panel is information only.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from core.strategy import scalp as scalp_mod
from core.strategy.library import CATALOGUE, resolve
from core.strategy.panel import PanelBuilder, build_panel, strategy_category, unavailable_entries
from core.strategy.scalp import ScalpConfig
from tests.unit.test_strategy_library import market

N_ALL = len(resolve(("all",)))
N_UNAVAILABLE = sum(1 for e in CATALOGUE if e.kind == "unavailable")


@pytest.fixture(scope="module")
def panel():
    return PanelBuilder(ScalpConfig(check_bars=1500)).build(market(1800, seed=3), "M15")


def test_one_reading_per_generator_plus_unavailable_entries(panel):
    avail = [r for r in panel.readings if r.available]
    unavail = [r for r in panel.readings if not r.available]
    assert len(avail) == N_ALL == panel.n_tested
    assert len(unavail) == N_UNAVAILABLE == len(unavailable_entries())
    assert len({r.name for r in avail}) == N_ALL


def test_counts_and_breadth_are_consistent(panel):
    ok = [r for r in panel.readings if r.available and r.error is None]
    assert panel.long_count == sum(r.signal > 0 for r in ok)
    assert panel.short_count == sum(r.signal < 0 for r in ok)
    assert panel.long_count + panel.short_count + panel.flat_count == len(ok)
    assert -1.0 <= panel.net_breadth <= 1.0
    assert -1.0 <= panel.weighted_breadth <= 1.0
    for cat, v in panel.by_category.items():
        assert v["long"] == sum(r.signal > 0 for r in ok if r.category == cat)
        assert v["short"] == sum(r.signal < 0 for r in ok if r.category == cat)


def test_proof_bar_is_bonferroni_over_all_strategies(panel):
    assert panel.required_z == pytest.approx(scalp_mod.required_z(N_ALL, 2.7))
    assert panel.required_z > 3.4


def test_readings_have_records_and_categories(panel):
    for r in panel.readings:
        if not r.available:
            continue
        assert r.category != "other", r.name
        assert r.signal in (-1, 0, 1)
        if r.edge is not None:
            assert -1.0 <= r.edge <= 1.0
            assert r.z == pytest.approx(r.edge * np.sqrt(r.calls))
    assert any(r.calls > 0 for r in panel.readings)


def test_random_walk_proves_nothing(panel):
    assert panel.proven == []


def test_json_serialisable_and_summary(panel):
    d = panel.to_dict()
    json.dumps(d)
    assert d["note"].startswith("information, not a trade signal")
    assert panel.summary().startswith("strategies M15:")


def test_a_raising_generator_is_reported_not_fatal(monkeypatch):
    def boom(df, atr, cfg):
        raise RuntimeError("broken rule")
    monkeypatch.setitem(scalp_mod.GENERATORS, "rsi2", boom)
    p = PanelBuilder(ScalpConfig(check_bars=600)).build(market(800, seed=1), "H1")
    bad = [r for r in p.readings if r.name == "rsi2"][0]
    assert bad.error and "broken rule" in bad.error
    assert bad.signal == 0 and not bad.proven
    assert len([r for r in p.readings if r.available]) == N_ALL


def test_cached_per_frame():
    b = PanelBuilder(ScalpConfig(check_bars=600))
    df = market(800, seed=2)
    p1 = b.build(df, "M15")
    t = time.perf_counter()
    p2 = b.build(df, "M15")
    assert p2 is p1 and time.perf_counter() - t < 0.05
    assert b.build(df.iloc[:-1], "M15") is not p1


def test_empty_or_bad_frames_give_an_empty_panel():
    import pandas as pd
    assert build_panel(pd.DataFrame(), "M15").readings == []
    bad = pd.DataFrame({"close": [1.0, 2.0]})
    assert build_panel(bad, "M15").readings == []


def test_reading_matches_truncated_frame():
    """The newest-bar call at t equals the call computed on the frame cut at t."""
    df = market(1200, seed=5)
    b = PanelBuilder(ScalpConfig(check_bars=600))
    full, _ = b.signals(df)
    for cut in (700, 950, 1199):
        p = PanelBuilder(ScalpConfig(check_bars=600)).build(df.iloc[:cut + 1], "M15")
        for r in p.readings:
            if r.available and r.error is None:
                assert r.signal == int(full[r.name][cut]), (r.name, cut)


def test_strategy_category_known():
    assert strategy_category("fvg") == "smc"
    assert strategy_category("definitely_not_a_strategy") == "other"


def test_full_library_speed():
    df = market(4500, seed=7)
    b = PanelBuilder()
    t = time.perf_counter()
    b.build(df, "M15")
    elapsed = time.perf_counter() - t
    print(f"full library panel on 4500 bars: {elapsed:.2f}s")
    assert elapsed < 8.0      # runs in a background thread in the live bot
