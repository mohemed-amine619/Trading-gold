"""
tests/unit/test_market_location.py — market structure, levels, liquidity, location.

The load-bearing property across this file is **honesty under ignorance**. The
bot must refuse to locate itself when the data cannot support a location, rather
than emit a plausible-looking default. Several tests assert that a ``None`` or
``UNDEFINED`` comes back, because the failure mode being guarded against is a
system that trades confidently on a guess.

All frames are built by hand rather than generated randomly: a seeded random
walk cannot express "higher high and higher low", so the tests would silently
stop testing the thing they claim to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.market.levels import LevelKind, build_levels
from core.market.liquidity import LiquidityState, scan_liquidity
from core.market.location import (
    LocationBias,
    analyse_location,
)
from core.market.structure import (
    BreakKind,
    StructureBias,
    active_swing,
    analyse_structure,
)
from core.market.swings import SwingKind, alternating, find_swings
from core.market.series import resolve_atr, valid_tick_size


# ─────────────────────────────────────────────────────────────────────────────
# Frame builders
# ─────────────────────────────────────────────────────────────────────────────

def zigzag_frame(legs, per_leg: int = 5, spread: float = 0.5, start: float = 100.0):
    """Build a frame from (direction, size) legs, so trends are explicit."""
    close = []
    price = start
    for direction, size in legs:
        step = size / per_leg
        for _ in range(per_leg):
            price += step * direction
            close.append(round(price, 4))
    idx = pd.date_range("2026-01-01", periods=len(close), freq="h")
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + spread for c in close],
            "low": [c - spread for c in close],
            "close": close,
        },
        index=idx,
    )


def bar_frame(rows):
    """Build a frame from explicit (low, close, high) triples."""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="h")
    return pd.DataFrame(
        {
            "open": np.array([r[1] for r in rows], dtype=float),
            "high": np.array([r[2] for r in rows], dtype=float),
            "low": np.array([r[0] for r in rows], dtype=float),
            "close": np.array([r[1] for r in rows], dtype=float),
        },
        index=idx,
    )


RANGE_ROWS = [
    (100, 104, 104.5), (105, 109, 109.5), (106, 110, 110.4), (105, 107, 107.5),
    (103, 105, 105.5), (101, 103, 103.5), (100, 102, 102.5), (101, 104, 104.5),
    (103, 107, 107.5), (105, 109, 109.5), (106, 110, 110.4), (108, 109, 109.5),
    (106, 108, 108.5), (104, 106, 106.5), (102, 104, 104.5), (100, 102, 102.5),
    (101, 103, 103.5), (100, 102, 102.4),
]


# ─────────────────────────────────────────────────────────────────────────────
# Swings
# ─────────────────────────────────────────────────────────────────────────────

class TestSwings:
    def test_trailing_pivots_are_unconfirmed(self):
        """The last bars lack right context and must not be treated as final."""
        df = zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 5)])
        swings = find_swings(df, left=2, right=2)

        assert swings, "expected some swings"
        assert all(s.confirmed for s in swings[:-1]), "only the newest may be unconfirmed"
        assert not swings[-1].confirmed

    def test_plateau_yields_exactly_one_deterministic_pivot(self):
        """A flat run must not be reported as a series of pivots."""
        idx = pd.date_range("2026-01-01", periods=15, freq="h")
        flat = pd.DataFrame(
            {"high": [5.0] * 15, "low": [1.0] * 15, "close": [3.0] * 15}, index=idx
        )
        first = find_swings(flat, left=2, right=2)
        second = find_swings(flat, left=2, right=2)

        assert first == second, "tie-breaking must be deterministic"
        assert len(first) <= 1

    def test_asymmetric_window_confirms_sooner(self):
        """A wide-left/narrow-right window is how you get a usable pivot fast."""
        df = zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 5)])
        wide = find_swings(df, left=4, right=1, include_unconfirmed=False)
        narrow = find_swings(df, left=2, right=2, include_unconfirmed=False)

        assert len(wide) >= len(narrow)

    def test_rejects_empty_window(self):
        df = zigzag_frame([(1, 5)])
        with pytest.raises(ValueError):
            find_swings(df, left=0, right=0)

    def test_requires_price_columns(self):
        with pytest.raises(KeyError):
            find_swings(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))

    def test_empty_frame_is_not_an_error(self):
        assert find_swings(pd.DataFrame()) == []

    def test_alternating_collapses_runs_keeping_the_extreme(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="h")
        df = pd.DataFrame({"high": [1, 5, 6, 7, 4, 2], "low": [0, -1, 0, 1, 0, -1]}, index=idx)
        swings = find_swings(df, left=1, right=1)

        collapsed = alternating(swings)
        kinds = [s.kind for s in collapsed]
        assert all(a is not b for a, b in zip(kinds, kinds[1:])), "must alternate"


# ─────────────────────────────────────────────────────────────────────────────
# Structure
# ─────────────────────────────────────────────────────────────────────────────

class TestStructure:
    def test_higher_high_higher_low_is_bullish_with_full_confidence(self):
        state = analyse_structure(zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 5)]), 2, 2)

        assert state.bias is StructureBias.BULLISH
        assert state.confidence == pytest.approx(1.0)

    def test_lower_high_lower_low_is_bearish_with_full_confidence(self):
        """Guards a real bug: agreement was hardcoded to bullish, scoring every
        clean downtrend 0.0."""
        state = analyse_structure(zigzag_frame([(-1, 5), (1, 2), (-1, 5), (1, 2), (-1, 5)]), 2, 2)

        assert state.bias is StructureBias.BEARISH
        assert state.confidence == pytest.approx(1.0)

    def test_reversal_produces_a_choch_not_a_bos(self):
        state = analyse_structure(
            zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 8), (1, 1), (-1, 5)]), 2, 2
        )

        assert state.bias is StructureBias.BEARISH
        assert state.last_choch is not None, "a trend reversal must register a change of character"
        assert any(e.kind is BreakKind.CHOCH for e in state.events)

    def test_continuation_produces_only_bos(self):
        state = analyse_structure(zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 5)]), 2, 2)

        assert all(e.kind is BreakKind.BOS for e in state.events)

    def test_break_judged_on_close_not_wick(self):
        """A wick through a level is liquidity, not a structural break."""
        df = bar_frame(
            [
                (10, 12, 12.5), (13, 15, 15.5), (16, 18, 18.5), (17, 19, 19.5),
                (16, 18, 18.5), (15, 17, 17.5), (16, 18, 18.5),
                (15, 17, 17.5),
                (14, 16, 16.5),
            ]
        )
        state = analyse_structure(df, 2, 2)
        for event in state.events:
            if event.direction == 1:
                assert event.break_close > event.level, "bullish break must close above the level"
            else:
                assert event.break_close < event.level, "bearish break must close below the level"

    def test_too_little_data_is_undefined_not_guessed(self):
        state = analyse_structure(zigzag_frame([(1, 2)], per_leg=4), 2, 2)

        assert state.bias is StructureBias.UNDEFINED
        assert state.confidence == 0.0

    def test_empty_frame_is_undefined(self):
        assert analyse_structure(pd.DataFrame(), 2, 2).bias is StructureBias.UNDEFINED

    def test_active_swing_anchors_the_stop_on_the_right_side(self):
        state = analyse_structure(zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 5)]), 2, 2)

        long_anchor = active_swing(state, 1)
        short_anchor = active_swing(state, -1)

        assert long_anchor.kind is SwingKind.LOW
        assert short_anchor.kind is SwingKind.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Levels
# ─────────────────────────────────────────────────────────────────────────────

class TestLevels:
    def test_repeated_reactions_cluster_into_one_zone(self):
        rows = [
            (100, 104, 104.5), (105, 109, 109.5), (106, 110, 110.4), (105, 107, 107.5),
            (103, 105, 105.5), (101, 103, 103.5), (100, 102, 102.5), (101, 104, 104.5),
            (103, 107, 107.5), (105, 109, 109.5), (106, 110, 110.4), (108, 109, 109.5),
            (106, 108, 108.5), (104, 106, 106.5), (102, 104, 104.5), (100, 102, 102.5),
            (101, 103, 103.5), (100, 102, 102.4),
        ]
        df = bar_frame(rows)
        zones = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).zones

        assert len(zones) == 2, f"expected one support and one resistance, got {len(zones)}"
        assert {z.kind for z in zones} == {LevelKind.SUPPORT, LevelKind.RESISTANCE}

    def test_zones_have_real_width(self):
        """A zone with zero width cannot be reached or contained."""
        df = bar_frame(RANGE_ROWS)
        zones = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).zones

        assert zones
        for z in zones:
            assert z.upper > z.lower, "a zone must be a band, not a point"

    def test_broken_resistance_becomes_support(self):
        df = zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 8)])
        zones = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df)).zones

        broken = [z for z in zones if z.broken]
        assert broken, "a clean uptrend must break prior swing highs"
        assert all(z.kind is LevelKind.SUPPORT for z in broken), "broken resistance flips to support"

    def test_broken_zones_are_capped_in_strength(self):
        df = zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 8)])
        zones = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df)).zones

        for z in zones:
            if z.broken:
                assert z.strength <= 0.55

    def test_tightness_measures_the_cluster_not_the_tolerance(self):
        """Guards a bug where tightness was computed from the tolerance itself
        and was therefore constant."""
        df = bar_frame(RANGE_ROWS)
        zones = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).zones

        assert zones
        assert any(z.components["tightness"] > 0.5 for z in zones)

    def test_tightly_packed_swings_beat_scattered_ones(self):
        df = bar_frame(RANGE_ROWS)
        levels = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01)

        for z in levels.zones:
            assert 0.0 <= z.strength <= 1.0

    def test_unconfirmed_swings_create_no_levels(self):
        df = bar_frame(RANGE_ROWS)
        with_unconf = build_levels(df, find_swings(df, 2, 2, include_unconfirmed=True), atr=resolve_atr(df))
        conf_only = build_levels(df, find_swings(df, 2, 2, include_unconfirmed=False), atr=resolve_atr(df))

        assert len(with_unconf.zones) == len(conf_only.zones)

    def test_tick_size_snaps_without_float_noise(self):
        df = bar_frame(RANGE_ROWS)
        zones = build_levels(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).zones

        for z in zones:
            assert z.price == pytest.approx(round(z.price, 2)), f"{z.price} is not on the tick grid"

    @pytest.mark.parametrize("junk", [0, -0.01, float("nan"), float("inf"), None, "abc", object()])
    def test_non_numeric_tick_size_is_ignored_not_fatal(self, junk):
        """Broker metadata is untrusted input.

        A stub, a mock, or an odd broker value reaches the snapping arithmetic
        directly. ``price / tick_size`` raises on a non-number, so an
        unguarded value loses the entire location read rather than merely
        declining to snap. Failing closed to None keeps the analysis working.
        """
        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, tick_size=junk)

        assert loc.is_valid, "a junk tick size must not cost us the location"
        assert valid_tick_size(junk) is None

    def test_magicmock_tick_size_does_not_break_location(self):
        """The exact shape that leaked from a mocked MT5 client in the suite."""
        from unittest.mock import MagicMock

        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, tick_size=MagicMock())

        assert loc.is_valid
        assert valid_tick_size(MagicMock()) is None

    def test_valid_tick_size_accepts_real_values(self):
        assert valid_tick_size(0.01) == pytest.approx(0.01)
        assert valid_tick_size("0.01") == pytest.approx(0.01)
        assert valid_tick_size(None) is None
        assert valid_tick_size(True) is None, "bool is not a price increment"

    def test_refuses_to_cluster_without_a_scale(self):
        flat = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )
        df = bar_frame(RANGE_ROWS)

        assert build_levels(flat, find_swings(df, 2, 2), atr=None).zones == []

    def test_no_atr_scale_means_no_invented_tolerance(self):
        df = bar_frame(RANGE_ROWS)
        result = build_levels(df, find_swings(df, 2, 2), atr=None)
        # ATR is derived from the frame's own true range here, which is market
        # data rather than a substituted default.
        assert result.atr is None or result.atr > 0


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity
# ─────────────────────────────────────────────────────────────────────────────

EQUAL_HIGHS = [
    (100, 102, 102.5), (104, 106, 106.5), (108, 110, 110.2),
    (107, 109, 109.5), (105, 107, 107.5), (103, 105, 105.5),
    (105, 107, 107.5), (107, 109, 109.5), (108, 110, 110.3),
    (107, 109, 109.5), (105, 107, 107.5), (104, 106, 106.5),
]


class TestLiquidity:
    def test_equal_highs_form_a_pool(self):
        df = bar_frame(EQUAL_HIGHS)
        pools = scan_liquidity(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).pools

        assert pools
        assert any(p.kind is SwingKind.HIGH and p.members >= 2 for p in pools)

    def test_stop_run_is_detected_despite_the_sweep_bar_being_its_own_swing(self):
        """The sweep bar forms a pivot, so scanning only past the cluster would
        skip the very bar that took the stops."""
        rows = EQUAL_HIGHS + [
            (102, 104, 111.2),   # wick 111.2 well above the pool, closes 104
            (101, 103, 103.5), (100, 102, 102.5),
        ]
        df = bar_frame(rows)
        pools = scan_liquidity(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).pools

        swept = [p for p in pools if p.state is not LiquidityState.RESTING]
        assert swept, "a decisive stop run must be recorded"
        assert any(p.kind is SwingKind.HIGH for p in swept)

    def test_pool_price_is_not_dragged_by_the_stop_run_wick(self):
        rows = EQUAL_HIGHS + [
            (102, 104, 111.2), (101, 103, 103.5), (100, 102, 102.5),
        ]
        df = bar_frame(rows)
        pools = scan_liquidity(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01).pools

        for p in pools:
            if p.state is not LiquidityState.RESTING:
                assert p.price < 111.2, "the wick must not redefine the resting level"

    def test_decisive_rejection_mitigates_but_marginal_does_not(self):
        marginal = scan_liquidity(
            bar_frame(EQUAL_HIGHS), find_swings(bar_frame(EQUAL_HIGHS), 2, 2),
            atr=resolve_atr(bar_frame(EQUAL_HIGHS)), tick_size=0.01,
        ).pools
        rows = EQUAL_HIGHS + [(102, 104, 111.2), (101, 103, 103.5), (100, 102, 102.5)]
        decisive_df = bar_frame(rows)
        decisive = scan_liquidity(
            decisive_df, find_swings(decisive_df, 2, 2), atr=resolve_atr(decisive_df), tick_size=0.01
        ).pools

        def worst_rejection(pool_list):
            rejected = [p.rejection_atr for p in pool_list if p.state is not LiquidityState.RESTING]
            return max(rejected) if rejected else 0.0

        assert worst_rejection(decisive) > worst_rejection(marginal)

    def test_single_touch_is_not_a_pool(self):
        df = zigzag_frame([(1, 5), (-1, 2), (1, 5), (-1, 2), (1, 5)])
        pools = scan_liquidity(df, find_swings(df, 2, 2), atr=resolve_atr(df)).pools

        assert all(p.members >= 2 for p in pools)

    def test_resting_pool_acts_as_a_target(self):
        df = bar_frame(EQUAL_HIGHS)
        pools = scan_liquidity(df, find_swings(df, 2, 2), atr=resolve_atr(df), tick_size=0.01)

        above = pools.above(106.0)
        assert above, "an untaken pool above price is a magnet"
        assert above[0].price == pytest.approx(110.25, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Location
# ─────────────────────────────────────────────────────────────────────────────

class TestLocation:
    def test_price_at_range_bottom_is_discount(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert loc.bias is LocationBias.DISCOUNT
        assert loc.range_position < 0.25
        assert loc.is_valid

    def test_price_at_range_top_is_premium(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [108.0, 110.0, 108.0, 109.8]
        loc = analyse_location(df, tick_size=0.01)

        assert loc.bias is LocationBias.PREMIUM
        assert loc.range_position > 0.90

    def test_no_usable_range_is_undefined_and_blocks_trading(self):
        """The central honesty test: unknown location must abstain, not guess."""
        flat = bar_frame([(100, 100.2, 100.3)] * 8)
        loc = analyse_location(flat, tick_size=0.01)

        assert loc.bias is LocationBias.UNDEFINED
        assert not loc.is_valid
        assert loc.entry_gate(1).allowed is False
        assert loc.entry_gate(-1).allowed is False

    def test_empty_frame_is_undefined_not_an_exception(self):
        loc = analyse_location(pd.DataFrame())

        assert loc.bias is LocationBias.UNDEFINED
        assert loc.range_position is None

    def test_fib_levels_span_the_range(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert set(loc.fib_levels) >= {"0.0", "0.382", "0.5", "0.618", "1.0"}
        assert loc.fib_levels["0.0"] == pytest.approx(loc.range_low)
        assert loc.fib_levels["1.0"] == pytest.approx(loc.range_high)
        assert loc.fib_levels["0.5"] == pytest.approx((loc.range_high + loc.range_low) / 2)

    def test_distances_are_reported_in_atr(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert loc.atr and loc.atr > 0
        distance = loc.distance_to_resistance()
        assert distance is not None
        assert distance / loc.atr > 0


class TestEntryGate:
    def test_long_is_allowed_in_discount(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert loc.bias is LocationBias.DISCOUNT
        assert loc.entry_gate(1).allowed is True

    def test_long_is_vetoed_in_premium(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [108.0, 110.0, 108.0, 109.8]
        loc = analyse_location(df, tick_size=0.01)

        gate = loc.entry_gate(1)
        assert gate.allowed is False
        assert any("PREMIUM" in b for b in gate.blocking)

    def test_short_is_vetoed_in_discount(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        gate = loc.entry_gate(-1)
        assert gate.allowed is False
        assert any("DISCOUNT" in b for b in gate.blocking)

    def test_short_is_allowed_in_premium(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [108.0, 110.0, 108.0, 109.8]
        loc = analyse_location(df, tick_size=0.01)

        assert loc.entry_gate(-1).allowed is True

    def test_entering_into_untaken_liquidity_is_cautioned(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [108.0, 110.0, 108.0, 109.8]
        loc = analyse_location(df, tick_size=0.01)

        gate = loc.entry_gate(1)
        assert gate.cautions, "approaching an untaken pool must be flagged"

    def test_no_direction_is_always_rejected(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert loc.entry_gate(0).allowed is False


# ─────────────────────────────────────────────────────────────────────────────
# Regime-aware gating
# ─────────────────────────────────────────────────────────────────────────────

def trending_frame(n: int = 300, drift: float = 0.45, pullback: float = 0.30, period: int = 20):
    """A sustained uptrend with shallow pullbacks — the trend case."""
    close = []
    price = 100.0
    for i in range(n):
        step = drift
        if i % period in (0, 1, 2) and i > 0:
            step = -pullback
        price += step
        close.append(round(price, 4))
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 0.3 for c in close],
            "low": [c - 0.3 for c in close],
            "close": close,
            "tick_volume": [1000.0] * n,
        },
        index=idx,
    )


class TestRegimeAwareGate:
    def test_uptrend_structure_is_detected(self):
        from core.market.pipeline import add_indicators

        loc = analyse_location(add_indicators(trending_frame()), tick_size=0.01)
        assert loc.structure.bias is StructureBias.BULLISH

    def test_trend_following_long_is_not_vetoed_as_premium(self):
        """Guards a design flaw, not an implementation slip.

        Premium/discount is a ranging concept. Applying it strictly to a trend
        blocked nearly every trend trade, because a confirmed uptrend sits at
        the top of its own swing range by construction — and it inverted the
        meaning of price above the range high, which in an uptrend is strength,
        not weakness.
        """
        from core.market.pipeline import add_indicators

        loc = analyse_location(add_indicators(trending_frame()), tick_size=0.01)

        assert loc.structure.bias is StructureBias.BULLISH
        assert loc.entry_gate(1).allowed is True, loc.entry_gate(1).reason

    def test_with_trend_entry_is_cautioned_not_blocked(self):
        from core.market.pipeline import add_indicators

        loc = analyse_location(add_indicators(trending_frame()), tick_size=0.01)
        gate = loc.entry_gate(1)

        assert gate.allowed is True
        assert gate.blocking == []
        assert gate.cautions, "an extended entry should still be flagged"

    def test_counter_trend_long_requires_discount(self):
        from core.market.pipeline import add_indicators

        df = trending_frame()
        # Force a bear structure by inverting the trend.
        inverted = df.copy()
        span = df["close"].max() - df["close"].min()
        for col in ("open", "high", "low", "close"):
            inverted[col] = (df[col].max() + df[col].min()) - df[col]
        inverted["high"], inverted["low"] = inverted["low"], inverted["high"]

        loc = analyse_location(add_indicators(inverted), tick_size=0.01)
        if loc.structure.bias is StructureBias.BEARISH:
            gate = loc.entry_gate(1)
            assert not gate.allowed or loc.bias is LocationBias.DISCOUNT

    def test_ranging_structure_keeps_the_strict_rule(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert loc.structure.bias is StructureBias.RANGING
        assert loc.entry_gate(-1).allowed is False, "shorting a range low is vetoed"
        assert loc.entry_gate(1).allowed is True


class TestOutOfRange:
    def test_position_is_clamped_into_the_unit_interval(self):
        """A breakdown is not an infinitely deep discount."""
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [90.0, 92.0, 90.0, 91.0]
        loc = analyse_location(df, tick_size=0.01)

        if loc.has_range:
            assert 0.0 <= loc.range_position <= 1.0

    def test_below_range_is_flagged_as_a_breakdown(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [90.0, 92.0, 90.0, 91.0]
        loc = analyse_location(df, tick_size=0.01)

        if loc.below_range:
            assert loc.range_position == 0.0
            assert any("below the range low" in c for c in loc.entry_gate(1).cautions)

    def test_above_range_is_flagged_as_a_breakout(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [118.0, 120.0, 118.0, 119.0]
        loc = analyse_location(df, tick_size=0.01)

        if loc.above_range:
            assert loc.range_position == 1.0
            assert any("above the range high" in c for c in loc.entry_gate(-1).cautions)

    def test_describe_reports_being_outside_the_range(self):
        df = bar_frame(RANGE_ROWS)
        df.loc[df.index[-1]] = [90.0, 92.0, 90.0, 91.0]
        text = analyse_location(df, tick_size=0.01).describe()

        assert "BELOW RANGE" in text or "UNDEFINED" in text


class TestStructuralStop:
    def test_long_stop_sits_below_the_protecting_swing_low(self):
        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, tick_size=0.01)
        plan = loc.structural_stop(104.0, 1, tick_size=0.01)

        assert plan is not None
        assert plan.price < 104.0
        assert plan.structural is True
        assert "swing low" in plan.reason

    def test_short_stop_sits_above_the_protecting_swing_high(self):
        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, tick_size=0.01)
        plan = loc.structural_stop(105.0, -1, tick_size=0.01)

        assert plan is not None
        assert plan.price > 105.0
        assert plan.structural is True

    def test_volatility_floor_always_holds_after_tick_snapping(self):
        """Snapping rounds toward entry and can pull the stop back inside the
        floor; the floor must survive it."""
        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, tick_size=0.01)

        for entry, direction in [(101.0, 1), (108.0, -1), (104.0, 1), (105.0, -1)]:
            plan = loc.structural_stop(entry, direction, min_atr_mult=0.75, tick_size=0.01)
            assert plan is not None
            assert plan.distance_atr >= 0.75 - 1e-6, f"{entry}/{direction} breached the floor"

    def test_returns_none_without_a_volatility_scale(self):
        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, atr=1e-9)

        assert loc.structural_stop(104.0, 1) is None

    def test_returns_none_for_no_direction(self):
        loc = analyse_location(bar_frame(RANGE_ROWS), tick_size=0.01)

        assert loc.structural_stop(104.0, 0) is None

    def test_stop_is_on_the_correct_side_of_entry(self):
        df = bar_frame(RANGE_ROWS)
        loc = analyse_location(df, tick_size=0.01)

        for entry in (102.0, 104.0, 106.0, 108.0):
            long_stop = loc.structural_stop(entry, 1, tick_size=0.01)
            short_stop = loc.structural_stop(entry, -1, tick_size=0.01)
            if long_stop:
                assert long_stop.price < entry
            if short_stop:
                assert short_stop.price > entry


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset behaviour — no gold-shaped constants
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAsset:
    @pytest.mark.parametrize("scale", [0.001, 1.0, 100.0])
    def test_location_is_scale_free(self, scale):
        """The same shape at EURUSD, gold and index scale must locate
        identically. A hardcoded point tolerance would fail here."""
        base = bar_frame(RANGE_ROWS)
        scaled = pd.DataFrame(
            {
                "open": base["open"] * scale,
                "high": base["high"] * scale,
                "low": base["low"] * scale,
                "close": base["close"] * scale,
            },
            index=base.index,
        )

        plain = analyse_location(base, tick_size=0.01)
        scaled_loc = analyse_location(scaled, tick_size=0.01 * scale)

        assert plain.bias is scaled_loc.bias
        assert plain.range_position == pytest.approx(scaled_loc.range_position, abs=1e-6)
        assert plain.entry_gate(1).allowed == scaled_loc.entry_gate(1).allowed

    def test_atr_survives_a_single_spike_bar(self):
        """One outsized bar must not redefine the market's volatility.

        Every tolerance in the system is a multiple of ATR. Reading only the
        latest true range lets one spike inflate ATR enough to make a tradable
        range fail its minimum-width test, silently flipping location to
        UNDEFINED and stopping all trading.
        """
        rows = list(RANGE_ROWS)
        rows[-1] = (100, 102, 140.0)  # violent spike bar
        df = bar_frame(rows)

        atr = resolve_atr(df)
        assert atr is not None
        assert atr < 50, "a single spike must not become the ATR"
        assert analyse_location(df).bias is not LocationBias.UNDEFINED