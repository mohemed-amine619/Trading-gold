"""
core/execution/states.py — Explicit order execution states.

Why this module exists
----------------------
The central safety problem in order handling is that "we did not get a
response" and "the broker rejected it" look identical if you model execution
as success/failure. They are not the same, and treating them the same causes
duplicate orders: a request times out, the caller assumes failure, retries,
and the first order fills anyway.

This module makes the ambiguous case explicit and unrepresentable-as-success.

    FILLED / PARTIALLY_FILLED / REJECTED / CANCELLED  → outcome known
    UNKNOWN                                          → outcome NOT known

``UNKNOWN`` is never a terminal success and never an automatic retry. It is a
signal to reconcile against the broker before deciding anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.mt5.types import RetCode

logger = logging.getLogger(__name__)


class ExecutionState(str, Enum):
    """Outcome of an order submission."""

    # ── Terminal, positive ─────────────────────────────────────────────────
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"

    # ── Terminal, negative ─────────────────────────────────────────────────
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    # ── Non-terminal / indeterminate ────────────────────────────────────────
    # The request may or may not have reached the broker. Reconcile required.
    UNKNOWN = "unknown"

    # ── Never sent ─────────────────────────────────────────────────────────
    # Safe to (re)submit: the broker demonstrably did not act on it.
    NOT_SENT = "not_sent"


#: States where the broker provably did not accept the order.
#:
#: A rejection means no position exists at the broker for this attempt, so it
#: is safe to resubmit — with a **new** idempotency key, since the original
#: decision has already been recorded. NOT_SENT covers the transient
#: price-refusal family (requote), where the same key may be retried safely.
INDETERMINATE_STATES = frozenset({ExecutionState.UNKNOWN})

#: States safe to resubmit: the broker provably did not act on the order.
SAFE_TO_RESUBMIT = frozenset({
    ExecutionState.NOT_SENT,
    ExecutionState.REJECTED,
    ExecutionState.EXPIRED,
})

#: States representing a completed fill (full or partial).
FILLED_STATES = frozenset({ExecutionState.FILLED, ExecutionState.PARTIALLY_FILLED})

#: Terminal states — the outcome will not change without a new action.
TERMINAL_STATES = frozenset({
    ExecutionState.FILLED,
    ExecutionState.PARTIALLY_FILLED,
    ExecutionState.REJECTED,
    ExecutionState.CANCELLED,
    ExecutionState.EXPIRED,
})

#: Broker retcodes whose outcome is genuinely unknown to us.
_UNKNOWN_RETCODE = frozenset({RetCode.TIMEOUT, RetCode.CONNECTION})

#: Retcodes that mean "broker did not accept the order" — safe to retry.
_NOT_SENT_RETCODE = frozenset({
    RetCode.REJECT,
    RetCode.NO_MONEY,
    RetCode.MARKET_CLOSED,
    RetCode.TRADE_DISABLED,
    RetCode.INVALID_STOPS,
    RetCode.INVALID_VOLUME,
    RetCode.INVALID_PRICE,
})


def state_from_retcode(
    retcode: int,
    volume_filled: float = 0.0,
    volume_requested: Optional[float] = None,
) -> ExecutionState:
    """
    Map a broker return code to an execution state.

    ``DONE`` means the broker accepted and processed the order, but some
    brokers report DONE even when fewer lots than requested were filled. Pass
    ``volume_requested`` to distinguish FILLED from PARTIALLY_FILLED; when it
    is omitted the retcode is trusted and the result is FILLED.
    """
    try:
        code = RetCode(int(retcode))
    except ValueError:
        # An unrecognised code is not evidence of success or failure.
        logger.warning("Unrecognised broker retcode %s; treating as UNKNOWN", retcode)
        return ExecutionState.UNKNOWN

    if code is RetCode.DONE:
        if (
            volume_requested is not None
            and volume_requested > 0
            and 0 < volume_filled < volume_requested - 1e-9
        ):
            return ExecutionState.PARTIALLY_FILLED
        return ExecutionState.FILLED
    if code is RetCode.PARTIAL_FILL:
        return ExecutionState.PARTIALLY_FILLED
    if code is RetCode.CANCEL:
        return ExecutionState.CANCELLED
    if code in _NOT_SENT_RETCODE:
        return ExecutionState.REJECTED
    if code in _UNKNOWN_RETCODE:
        return ExecutionState.UNKNOWN

    # Requote / price-changed family: the broker explicitly refused this
    # attempt and is waiting for a new price. Nothing is pending at the broker.
    if code in (RetCode.REQUOTE, RetCode.PRICE_CHANGED, RetCode.PRICE_OFF):
        return ExecutionState.NOT_SENT

    return ExecutionState.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionReport:
    """
    Result of one order submission attempt.

    Deliberately not a bare bool: the caller must confront *why* something
    happened, especially the difference between REJECTED and UNKNOWN.
    """

    state: ExecutionState
    idempotency_key: str
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    retcode: int = 0
    comment: str = ""
    attempts: int = 1

    # ── Safety predicates ───────────────────────────────────────────────────

    @property
    def is_filled(self) -> bool:
        return self.state in FILLED_STATES

    @property
    def is_indeterminate(self) -> bool:
        """
        True when the outcome is unknown.

        The ONLY safe response is to reconcile with the broker. Never retry.
        """
        return self.state is ExecutionState.UNKNOWN

    @property
    def is_safe_to_resubmit(self) -> bool:
        """True only when the broker provably did not accept the order."""
        return self.state in SAFE_TO_RESUBMIT

    @property
    def requires_reconciliation(self) -> bool:
        return self.state in INDETERMINATE_STATES or self.state in FILLED_STATES

    def summary(self) -> str:
        return (
            f"{self.state.value} (key={self.idempotency_key} "
            f"retcode={self.retcode} attempts={self.attempts})"
        )