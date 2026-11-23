"""
tests/unit/test_position_scratch.py — the scratch exit must cut trades that
never went the predicted way, and must NEVER cut a trade that did.

The second half is the safety-critical one. Winners in this account already
keep a median 25% of their peak; a scratch rule that fired on a pullback would
make the largest measured problem worse. Hence: judged on PEAK, never current.
"""

from __future__ import annotations

import pytest

from core.position.scratch import should_scratch


ATR = 100.0
AFTER = 600.0          # 10 minutes
MIN_PEAK = 0.3         # ATR


def scratch(peak, elapsed, *, atr=ATR, after=AFTER, min_peak=MIN_PEAK):
    return should_scratch(peak, atr, elapsed, after, min_peak)


# ─────────────────────────────────────────────────────────────────────────────
# Off by default
# ─────────────────────────────────────────────────────────────────────────────

class TestDisabledByDefault:
    @pytest.mark.parametrize("after", [0.0, 0, -1.0, None])
    def test_a_non_positive_budget_disables_the_rule(self, after):
        """An untouched configuration must change nothing."""
        assert scratch(0.0, 99999.0, after=after) is False

    def test_a_non_positive_threshold_disables_the_rule(self):
        """With no bar to clear, nothing can fail to clear it."""
        assert scratch(0.0, 99999.0, min_peak=0.0) is False
        assert scratch(0.0, 99999.0, min_peak=-0.3) is False


# ─────────────────────────────────────────────────────────────────────────────
# The case it exists for
# ─────────────────────────────────────────────────────────────────────────────

class TestScratchesTradesThatNeverWentTheRightWay:
    def test_never_in_profit_past_the_budget_is_scratched(self):
        """12 of 23 trades peaked below 0.3 ATR; all 12 lost ~2.12 each."""
        assert scratch(0.0, AFTER) is True

    def test_a_trade_that_only_went_against_us_is_scratched(self):
        """A negative peak is the clearest instance of 'never got going'."""
        assert scratch(-50.0, AFTER) is True

    def test_a_peak_just_below_the_bar_is_scratched(self):
        assert scratch(0.29 * ATR, AFTER) is True

    def test_exactly_at_the_budget_counts_as_elapsed(self):
        assert scratch(0.0, AFTER) is True


# ─────────────────────────────────────────────────────────────────────────────
# What it must never do
# ─────────────────────────────────────────────────────────────────────────────

class TestNeverCutsATradeThatWorked:
    def test_a_peak_at_the_bar_is_not_scratched(self):
        assert scratch(0.30 * ATR, AFTER) is False

    def test_a_winner_in_a_pullback_is_never_scratched(self):
        """THE safety test. A trade that ran to +0.8 ATR and is now at -0.1 is a
        GIVEBACK, owned by profit_giveback_pct and the trailing logic. Judging
        on current profit here would cut winners mid-pullback, which is already
        this account's biggest measured leak."""
        assert scratch(0.80 * ATR, 99999.0) is False

    def test_a_large_peak_is_never_scratched_however_long_it_runs(self):
        assert scratch(3.0 * ATR, 86400.0) is False

    def test_before_the_budget_nothing_is_scratched(self):
        """A trade must be given its time before being judged."""
        assert scratch(0.0, AFTER - 1) is False
        assert scratch(0.0, 0.0) is False


# ─────────────────────────────────────────────────────────────────────────────
# Scale
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdScalesWithAtr:
    def test_the_bar_is_an_atr_multiple_not_a_fixed_distance(self):
        """Multi-asset mandate: BTC ATR ~150, EURUSD ATR ~0.0004. A fixed point
        count cannot serve both."""
        # Same 0.2-ATR peak on two very different instruments: both scratched.
        assert should_scratch(0.2 * 150.0, 150.0, AFTER, AFTER, 0.3) is True
        assert should_scratch(0.2 * 0.0004, 0.0004, AFTER, AFTER, 0.3) is True
        # Same 0.5-ATR peak on both: neither scratched.
        assert should_scratch(0.5 * 150.0, 150.0, AFTER, AFTER, 0.3) is False
        assert should_scratch(0.5 * 0.0004, 0.0004, AFTER, AFTER, 0.3) is False


# ─────────────────────────────────────────────────────────────────────────────
# Fail closed
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownInputsRefuse:
    @pytest.mark.parametrize("atr", [None, 0.0, -1.0, float("nan"), float("inf"), "x", True])
    def test_an_unusable_atr_refuses(self, atr):
        assert should_scratch(0.0, atr, AFTER, AFTER, MIN_PEAK) is False

    @pytest.mark.parametrize("elapsed", [None, float("nan"), float("inf"), "x", True])
    def test_an_unusable_elapsed_refuses(self, elapsed):
        assert should_scratch(0.0, ATR, elapsed, AFTER, MIN_PEAK) is False

    @pytest.mark.parametrize("peak", [None, float("nan"), float("inf"), "x", True])
    def test_an_unusable_peak_refuses(self, peak):
        assert should_scratch(peak, ATR, AFTER, AFTER, MIN_PEAK) is False

    @pytest.mark.parametrize("after", [float("nan"), float("inf"), "x", True])
    def test_an_unusable_budget_refuses(self, after):
        assert should_scratch(0.0, ATR, AFTER, after, MIN_PEAK) is False

    @pytest.mark.parametrize("min_peak", [None, float("nan"), float("inf"), "x", True])
    def test_an_unusable_threshold_refuses(self, min_peak):
        assert should_scratch(0.0, ATR, AFTER, AFTER, min_peak) is False

    def test_it_never_raises(self):
        """Called once per second inside the management loop; an exception here
        would take down position management."""
        for args in [
            (None, None, None, None, None),
            ("a", "b", "c", "d", "e"),
            (object(), object(), object(), object(), object()),
        ]:
            assert should_scratch(*args) is False
