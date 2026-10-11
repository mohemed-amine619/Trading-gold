"""
core/execution/idempotency.py — Duplicate-order prevention.

The problem
-----------
An order submission can fail in a way that tells you nothing: the network
drops, the terminal hangs, the response never arrives. The broker may have
filled it. If the caller simply retries, the account now holds two positions
for one decision.

The mechanism
-------------
Every submission carries a caller-supplied ``idempotency_key``. Before sending,
the engine registers the key. A key that is already registered is never sent
again — the recorded report is returned instead, and if that report is
``UNKNOWN`` the caller is told to reconcile rather than resubmit.

This is deliberately in-process and conservative: it protects against the
retry loop creating duplicates, which is the dominant real-world cause. It is
not a distributed store; a restart loses the cache, which is why the caller
must reconcile with the broker after a restart before trading again.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from core.execution.states import ExecutionReport, ExecutionState

logger = logging.getLogger(__name__)


@dataclass
class _Record:
    report: ExecutionReport
    in_flight: bool = False


class IdempotencyLedger:
    """
    Thread-safe record of submitted orders, keyed by idempotency key.

    Guarantees: a given key is submitted to the broker **at most once**.
    """

    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def begin(self, key: str) -> bool:
        """
        Attempt to claim ``key`` for submission.

        Returns True when the caller may proceed to send.
        Returns False when the key was already used — do NOT send.
        """
        if not key:
            raise ValueError("idempotency_key is required")
        with self._lock:
            if key in self._records:
                logger.warning(
                    "Duplicate submission blocked for key=%s (state=%s)",
                    key, self._records[key].report.state.value,
                )
                return False
            self._records[key] = _Record(
                report=ExecutionReport(
                    state=ExecutionState.UNKNOWN,
                    idempotency_key=key,
                ),
                in_flight=True,
            )
            return True

    def complete(self, report: ExecutionReport) -> None:
        """Record the final outcome for a key."""
        with self._lock:
            self._records[report.idempotency_key] = _Record(
                report=report, in_flight=False
            )

    def get(self, key: str) -> Optional[ExecutionReport]:
        with self._lock:
            record = self._records.get(key)
            return record.report if record else None

    def is_in_flight(self, key: str) -> bool:
        with self._lock:
            record = self._records.get(key)
            return bool(record and record.in_flight)

    def needs_reconciliation(self, key: str) -> bool:
        """True when this key's outcome is still unknown to us."""
        with self._lock:
            record = self._records.get(key)
            return bool(
                record
                and (record.in_flight or record.report.is_indeterminate)
            )

    def unresolved_keys(self) -> list[str]:
        """Every key whose outcome still requires broker reconciliation."""
        with self._lock:
            return [
                k for k, r in self._records.items()
                if r.in_flight or r.report.is_indeterminate
            ]

    def clear(self) -> None:
        """
        Drop all records.

        Only valid after the caller has reconciled with the broker — otherwise
        this discards the very protection that prevents duplicates.
        """
        with self._lock:
            self._records.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Key generation
# ─────────────────────────────────────────────────────────────────────────────

def make_idempotency_key(prefix: str, *parts: object) -> str:
    """
    Build a deterministic key from a decision's identity.

    Deterministic (not random) so that a replayed decision — same signal, same
    symbol, same bar — produces the same key and is therefore caught as a
    duplicate rather than placed twice.
    """
    return ":".join([prefix, *(str(p) for p in parts)])