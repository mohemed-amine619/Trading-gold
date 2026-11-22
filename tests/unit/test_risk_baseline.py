"""tests/unit/test_risk_baseline.py — deposit must not read as daily profit."""

from __future__ import annotations

import pytest

from core.risk.baseline import (
    deposit_baseline_adjustment,
    withdrawal_peak_reduction,
)


FLAT = (390.04, 500.0, 500.0, 0, 0)


def test_deposit_on_a_flat_account_rebases_the_baseline():
    assert deposit_baseline_adjustment(*FLAT) == pytest.approx(109.96)


def test_rebasing_is_idempotent_once_applied():
    # After the baseline has been moved to the deposited equity, a second
    # connect must not add the deposit again.
    assert deposit_baseline_adjustment(500.0, 500.0, 500.0, 0, 0) == 0.0


def test_a_profitable_trade_is_not_a_deposit():
    # Same equity jump, but one trade was made today: profit and cash inflow
    # are indistinguishable, so the baseline must stay put.
    assert deposit_baseline_adjustment(390.04, 500.0, 500.0, 1, 0) == 0.0


def test_an_open_position_blocks_the_adjustment():
    assert deposit_baseline_adjustment(390.04, 500.0, 500.0, 0, 1) == 0.0


def test_withdrawal_never_lowers_the_baseline():
    assert deposit_baseline_adjustment(390.04, 300.0, 300.0, 0, 0) == 0.0


def test_no_change_is_no_adjustment():
    assert deposit_baseline_adjustment(390.04, 390.04, 390.04, 0, 0) == 0.0


def test_floating_equity_above_cash_balance_uses_the_smaller_figure():
    # Should not happen while flat, but an inconsistent snapshot must
    # understate rather than overstate the adjustment.
    assert deposit_baseline_adjustment(390.04, 520.0, 500.0, 0, 0) == pytest.approx(109.96)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 500.0, 500.0, 0, 0),          # unknown baseline
        (None, 500.0, 500.0, 0, 0),
        (390.04, None, 500.0, 0, 0),
        (390.04, 500.0, None, 0, 0),
        (390.04, 500.0, 500.0, None, 0),
        (390.04, 500.0, 500.0, 0, None),
        (float("nan"), 500.0, 500.0, 0, 0),
        (390.04, float("inf"), 500.0, 0, 0),
    ],
)
def test_unknown_inputs_fail_closed(args):
    assert deposit_baseline_adjustment(*args) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Withdrawal: the peak must follow cash out, or the bot halts for good.
# ─────────────────────────────────────────────────────────────────────────────

class TestWithdrawalPeakReduction:
    def test_the_live_case_a_withdrawal_relieves_the_halt(self):
        """Observed live: peak 1504.37, equity 503.07 after a 1001.30
        withdrawal reads as 66.6% total drawdown against a 20% ceiling and
        hard-halts every symbol until a manual reset."""
        reduction = withdrawal_peak_reduction(1504.37, 503.07, 1001.30)
        new_peak = 1504.37 - reduction
        assert new_peak == pytest.approx(503.07, abs=0.01)
        assert (new_peak - 503.07) / new_peak < 0.20, "halt must be relieved"

    def test_the_peak_never_falls_below_current_equity(self):
        """A withdrawal larger than the gap must not invent a profit."""
        reduction = withdrawal_peak_reduction(1000.0, 900.0, 5000.0)
        assert reduction == pytest.approx(100.0)
        assert 1000.0 - reduction == pytest.approx(900.0)

    def test_a_partial_withdrawal_leaves_the_remaining_drawdown(self):
        """Taking out 100 of a 300 shortfall must not erase the other 200."""
        reduction = withdrawal_peak_reduction(1000.0, 700.0, 100.0)
        assert reduction == pytest.approx(100.0)
        assert 1000.0 - reduction == pytest.approx(900.0)

    def test_no_withdrawal_changes_nothing(self):
        assert withdrawal_peak_reduction(1000.0, 700.0, 0.0) == 0.0

    def test_a_deposit_never_moves_the_peak(self):
        """Only withdrawals lower it. A positive balance delta reaches this
        function as 0.0 withdrawn; raising the peak would manufacture fresh
        drawdown headroom out of deposited cash."""
        assert withdrawal_peak_reduction(1000.0, 1200.0, 0.0) == 0.0

    def test_equity_at_or_above_the_peak_has_no_drawdown_to_relieve(self):
        assert withdrawal_peak_reduction(1000.0, 1000.0, 50.0) == 0.0
        assert withdrawal_peak_reduction(1000.0, 1500.0, 50.0) == 0.0

    @pytest.mark.parametrize("peak,equity,withdrawn", [
        (None, 500.0, 100.0), (1000.0, None, 100.0), (1000.0, 500.0, None),
        (float("nan"), 500.0, 100.0), (1000.0, float("inf"), 100.0),
        (1000.0, 500.0, float("nan")),
        (0.0, 500.0, 100.0), (-5.0, 500.0, 100.0),
        (1000.0, -1.0, 100.0), (1000.0, 500.0, -100.0),
        ("x", 500.0, 100.0), (1000.0, 500.0, "x"), (True, 500.0, 100.0),
    ])
    def test_unknown_or_nonsense_inputs_fail_closed(self, peak, equity, withdrawn):
        """An unreadable number must never move a safety reference."""
        assert withdrawal_peak_reduction(peak, equity, withdrawn) == 0.0
