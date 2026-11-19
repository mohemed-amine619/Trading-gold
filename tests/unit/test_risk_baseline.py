"""tests/unit/test_risk_baseline.py — deposit must not read as daily profit."""

from __future__ import annotations

import pytest

from core.risk.baseline import deposit_baseline_adjustment


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
