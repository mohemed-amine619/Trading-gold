"""
tests/unit/test_execution_safety.py — Duplicate prevention and unknown outcomes.

These tests encode two rules that are easy to violate and expensive to get
wrong in production:

1. **The same idempotency key is never submitted twice.**
2. **An indeterminate broker outcome is never retried.** A timeout does not
   prove the order failed — it may have filled. Retrying it is how accounts
   end up with duplicate positions.

No test here places a real order. A scripted fake adapter records every
request so assertions are made on what actually reached the "broker".
"""

from __future__ import annotations

import pytest

from core.execution.engine import ExecutionEngine
from core.execution.idempotency import IdempotencyLedger, make_idempotency_key
from core.execution.states import (
    ExecutionState,
    state_from_retcode,
)
from core.mt5.types import OrderRequest, OrderResult, RetCode, TradeAction, OrderType


# ─────────────────────────────────────────────────────────────────────────────
# Fake broker
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedAdapter:
    """Adapter returning a scripted sequence of results, recording all sends."""

    def __init__(self, results):
        self.results = list(results)
        self.sent: list[OrderRequest] = []

    def send_order(self, request: OrderRequest) -> OrderResult:
        self.sent.append(request)
        result = self.results.pop(0) if self.results else self.results
        if result is None:
            raise AssertionError("adapter called more times than the script allowed")
        return result

    def get_positions(self, symbol=None, magic=None):
        return []


def _request(volume: float = 0.1) -> OrderRequest:
    return OrderRequest(
        symbol="EURUSD",
        action=TradeAction.DEAL,
        type=OrderType.BUY,
        volume=volume,
        price=0.0,
        sl=0.0,
        tp=0.0,
        magic=888999,
        comment="test",
    )


def _done(volume: float = 0.1) -> OrderResult:
    return OrderResult(retcode=RetCode.DONE, deal=555, order=444, volume=volume, price=1.10)


# ─────────────────────────────────────────────────────────────────────────────
# Retcode -> state mapping
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("retcode,expected", [
    (RetCode.DONE, ExecutionState.FILLED),
    (RetCode.PARTIAL_FILL, ExecutionState.PARTIALLY_FILLED),
    (RetCode.REJECT, ExecutionState.REJECTED),
    (RetCode.CANCEL, ExecutionState.CANCELLED),
    (RetCode.INVALID_STOPS, ExecutionState.REJECTED),
    (RetCode.INVALID_VOLUME, ExecutionState.REJECTED),
    (RetCode.NO_MONEY, ExecutionState.REJECTED),
    (RetCode.MARKET_CLOSED, ExecutionState.REJECTED),
    (RetCode.REQUOTE, ExecutionState.NOT_SENT),
    (RetCode.PRICE_CHANGED, ExecutionState.NOT_SENT),
    # The critical mapping: these must never look like a clean failure.
    (RetCode.TIMEOUT, ExecutionState.UNKNOWN),
    (RetCode.CONNECTION, ExecutionState.UNKNOWN),
    (999999, ExecutionState.UNKNOWN),          # unrecognised -> unknown
])


def test_state_from_retcode(retcode, expected):
    assert state_from_retcode(retcode) == expected


def test_partial_fill_detected_when_less_volume_than_requested():
    """Some brokers report DONE even when the fill is partial."""
    assert state_from_retcode(
        RetCode.DONE, volume_filled=0.04, volume_requested=0.10
    ) == ExecutionState.PARTIALLY_FILLED


def test_done_with_full_volume_is_filled():
    assert state_from_retcode(
        RetCode.DONE, volume_filled=0.10, volume_requested=0.10
    ) == ExecutionState.FILLED


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency ledger
# ─────────────────────────────────────────────────────────────────────────────

def test_ledger_blocks_second_use_of_key():
    ledger = IdempotencyLedger()
    assert ledger.begin("k1") is True
    assert ledger.begin("k1") is False


def test_ledger_requires_a_key():
    ledger = IdempotencyLedger()
    with pytest.raises(ValueError):
        ledger.begin("")


def test_ledger_tracks_unresolved_keys():
    ledger = IdempotencyLedger()
    ledger.begin("k1")                      # in flight, outcome unknown
    assert ledger.is_in_flight("k1")
    assert ledger.needs_reconciliation("k1")
    assert "k1" in ledger.unresolved_keys()


def test_ledger_marks_completed_report_as_resolved():
    ledger = IdempotencyLedger()
    ledger.begin("k1")
    ledger.complete(
        __import__("core.execution.states", fromlist=["ExecutionReport"])
        .ExecutionReport(state=ExecutionState.FILLED, idempotency_key="k1")
    )
    assert ledger.unresolved_keys() == []


def test_make_idempotency_key_is_deterministic():
    a = make_idempotency_key("sig", "XAUUSD", "2026-01-01T10:15", 1)
    b = make_idempotency_key("sig", "XAUUSD", "2026-01-01T10:15", 1)
    assert a == b, "same decision must produce the same key, or duplicates slip through"
    assert a != make_idempotency_key("sig", "XAUUSD", "2026-01-01T10:15", -1)


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate prevention through the engine
# ─────────────────────────────────────────────────────────────────────────────

def test_identical_key_submits_only_once():
    """The core guarantee: one decision, one order."""
    adapter = ScriptedAdapter([_done()])
    engine = ExecutionEngine(adapter, max_retries=3, retry_delay_ms=0)

    first = engine.submit(_request(), "decision-1")
    second = engine.submit(_request(), "decision-1")

    assert len(adapter.sent) == 1, "adapter must be called exactly once"
    assert first.state == ExecutionState.FILLED
    assert second.state == ExecutionState.FILLED
    assert second.deal == first.deal


def test_duplicate_after_unknown_returns_original_report():
    adapter = ScriptedAdapter([OrderResult(retcode=RetCode.TIMEOUT)])
    engine = ExecutionEngine(adapter, max_retries=3, retry_delay_ms=0)

    first = engine.submit(_request(), "decision-x")
    second = engine.submit(_request(), "decision-x")

    assert len(adapter.sent) == 1
    assert first.is_indeterminate
    assert second.is_indeterminate


def test_distinct_keys_both_submit():
    adapter = ScriptedAdapter([_done(), _done()])
    engine = ExecutionEngine(adapter, retry_delay_ms=0)

    engine.submit(_request(), "decision-a")
    engine.submit(_request(), "decision-b")

    assert len(adapter.sent) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Unknown outcomes must not be retried
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("retcode", [RetCode.TIMEOUT, RetCode.CONNECTION])
def test_unknown_outcome_is_not_retried(retcode):
    """
    A timeout may mean the order filled. Retrying creates a duplicate, so the
    engine must stop after the first attempt and report UNKNOWN.
    """
    adapter = ScriptedAdapter([OrderResult(retcode=retcode)])
    engine = ExecutionEngine(adapter, max_retries=5, retry_delay_ms=0)

    report = engine.submit(_request(), "ambiguous-1")

    assert len(adapter.sent) == 1, "must not retry an indeterminate outcome"
    assert report.state == ExecutionState.UNKNOWN
    assert report.is_indeterminate is True
    assert report.is_safe_to_resubmit is False
    assert report.requires_reconciliation is True


def test_requote_is_retried_because_broker_refused():
    """
    Contrasting case: a requote means the broker explicitly declined this
    attempt and nothing is pending, so a bounded retry is safe.
    """
    adapter = ScriptedAdapter([
        OrderResult(retcode=RetCode.REQUOTE),
        _done(),
    ])
    engine = ExecutionEngine(adapter, max_retries=3, retry_delay_ms=0)

    report = engine.submit(_request(), "requote-1")

    assert len(adapter.sent) == 2
    assert report.state == ExecutionState.FILLED
    assert report.attempts == 2


def test_retry_budget_is_bounded():
    adapter = ScriptedAdapter([OrderResult(retcode=RetCode.REQUOTE)] * 10)
    engine = ExecutionEngine(adapter, max_retries=2, retry_delay_ms=0)

    report = engine.submit(_request(), "requote-loop")

    assert len(adapter.sent) == 3, "1 initial attempt + 2 retries"
    assert report.attempts == 3


def test_rejection_is_terminal_and_not_retried():
    adapter = ScriptedAdapter([OrderResult(retcode=RetCode.INVALID_STOPS, comment="too close")])
    engine = ExecutionEngine(adapter, max_retries=3, retry_delay_ms=0)

    report = engine.submit(_request(), "rejected-1")

    assert len(adapter.sent) == 1
    assert report.state == ExecutionState.REJECTED
    assert report.is_safe_to_resubmit is True


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

def test_engine_exposes_unresolved_keys_for_reconciliation():
    adapter = ScriptedAdapter([OrderResult(retcode=RetCode.TIMEOUT)])
    engine = ExecutionEngine(adapter, retry_delay_ms=0)

    engine.submit(_request(), "needs-check")

    assert engine.ledger.unresolved_keys() == ["needs-check"]
    assert engine.ledger.needs_reconciliation("needs-check") is True


def test_filled_order_is_not_left_pending_reconciliation():
    adapter = ScriptedAdapter([_done()])
    engine = ExecutionEngine(adapter, retry_delay_ms=0)

    engine.submit(_request(), "settled")

    assert engine.ledger.unresolved_keys() == []


def test_filled_report_requires_no_resubmission():
    report = __import__("core.execution.states", fromlist=["ExecutionReport"]).ExecutionReport(
        state=ExecutionState.FILLED, idempotency_key="k"
    )
    assert report.is_safe_to_resubmit is False
    assert report.is_filled is True