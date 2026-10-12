"""
core/execution/engine.py — Execution Engine.

Handles the order lifecycle:
- Sends orders via MT5Adapter
- Retries on requotes or transient errors
- Handles error reporting
"""

import logging
import time
from typing import Optional

from core.mt5.adapter import MT5Adapter
from core.mt5.types import OrderRequest, OrderResult, RetCode, TradeAction, OrderType
from core.execution.idempotency import IdempotencyLedger
from core.execution.states import (
    FILLED_STATES,
    ExecutionReport,
    ExecutionState,
    state_from_retcode,
)
from core.risk.engine import TradeProposal

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, adapter: MT5Adapter, max_retries: int = 3, retry_delay_ms: int = 500):
        self.adapter = adapter
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        # Duplicate-order protection. Shared across calls on this engine.
        self.ledger = IdempotencyLedger()

    def execute_proposal(
        self,
        proposal: TradeProposal,
        lot_size: float,
        magic: int = 888999,
        idempotency_key: Optional[str] = None,
    ) -> Optional[OrderResult]:
        """
        Execute an approved TradeProposal as a market order.
        If idempotency_key is provided, routes through submit() to prevent duplicate orders.
        """
        if lot_size <= 0:
            logger.error(f"Cannot execute {proposal.symbol} with lot_size <= 0")
            return None

        # BUY = 0, SELL = 1
        order_type = OrderType.BUY if proposal.direction == 1 else OrderType.SELL
        
        request = OrderRequest(
            symbol=proposal.symbol,
            action=TradeAction.DEAL,
            type=order_type,
            volume=lot_size,
            price=0.0,  # Market order
            sl=proposal.stop_loss,
            tp=proposal.take_profit,
            magic=magic,
            comment=f"{proposal.strategy_name[:20]}"
        )
        
        if idempotency_key:
            report = self.submit(request, idempotency_key=idempotency_key)
            return OrderResult(
                retcode=report.retcode if report.retcode is not None else (RetCode.DONE if report.state in FILLED_STATES else RetCode.REJECT),
                deal=report.deal or 0,
                order=report.order or 0,
                volume=report.volume or 0.0,
                price=report.price or 0.0,
                comment=report.comment or report.state.value,
            )

        return self._send_with_retry(request)

    def submit_proposal(
        self,
        proposal: TradeProposal,
        lot_size: float,
        idempotency_key: str,
        magic: int = 888999,
    ) -> ExecutionReport:
        """
        Submit an approved TradeProposal guarded by the IdempotencyLedger.
        Guarantees that a proposal with the same idempotency key is submitted at most once.
        """
        if lot_size <= 0:
            logger.error(f"Cannot submit {proposal.symbol} with lot_size <= 0")
            return ExecutionReport(
                state=ExecutionState.REJECTED,
                idempotency_key=idempotency_key,
                comment="lot_size <= 0",
            )

        order_type = OrderType.BUY if proposal.direction == 1 else OrderType.SELL
        request = OrderRequest(
            symbol=proposal.symbol,
            action=TradeAction.DEAL,
            type=order_type,
            volume=lot_size,
            price=0.0,
            sl=proposal.stop_loss,
            tp=proposal.take_profit,
            magic=magic,
            comment=f"{proposal.strategy_name[:20]}",
        )
        return self.submit(request, idempotency_key=idempotency_key)

    def close_position(self, ticket: int, symbol: str, volume: float, direction: int) -> Optional[OrderResult]:
        """
        Close an existing position by sending an opposite market order.
        """
        order_type = OrderType.SELL if direction == 1 else OrderType.BUY
        
        request = OrderRequest(
            symbol=symbol,
            action=TradeAction.DEAL,
            type=order_type,
            volume=volume,
            price=0.0,
            magic=888999,
            comment="close_pos"
        )
        
        # Set position ticket for MT5 if needed (supported in some brokers)
        # request.position = ticket (would require adding position field to OrderRequest)
        
        return self._send_with_retry(request)
        
    def modify_position(self, ticket: int, symbol: str, new_sl: float, new_tp: float) -> Optional[OrderResult]:
        """
        Modify Stop Loss and Take Profit of an existing position.
        Safety gate: never allows loosening an existing stop loss.
        """
        # Safety gate: query existing position and refuse to loosen stop loss
        try:
            positions = self.adapter.get_positions(symbol=symbol)
            pos = next((p for p in positions if p.ticket == ticket), None)
            if pos is not None and pos.sl > 0.0 and new_sl > 0.0:
                # BUY: direction == 1 (type == 0). Loosening means new_sl < pos.sl
                if pos.direction == 1 and new_sl < pos.sl:
                    logger.error(
                        "Safety violation blocked: attempting to widen/loosen BUY SL for ticket %d (current=%f, proposed=%f)",
                        ticket, pos.sl, new_sl,
                    )
                    return None
                # SELL: direction == -1 (type == 1). Loosening means new_sl > pos.sl
                elif pos.direction == -1 and new_sl > pos.sl:
                    logger.error(
                        "Safety violation blocked: attempting to widen/loosen SELL SL for ticket %d (current=%f, proposed=%f)",
                        ticket, pos.sl, new_sl,
                    )
                    return None
        except Exception as exc:
            logger.warning("Could not verify existing position SL during modify_position: %s", exc)

        request = OrderRequest(
            symbol=symbol,
            action=TradeAction.SLTP,
            type=OrderType.BUY, # Ignored for SLTP
            volume=0.0,         # Ignored for SLTP
            price=0.0,
            sl=new_sl,
            tp=new_tp,
            magic=888999,
            comment="modify_sltp",
            position=ticket
        )
        
        return self._send_with_retry(request)

    def _send_with_retry(self, request: OrderRequest) -> OrderResult:
        """
        Legacy raw-result submission. Retries only on explicit requote.

        Retries are attempted ONLY when the broker definitively refused the
        attempt (requote / price changed). An indeterminate outcome
        (timeout, connection loss) is returned as-is: the order may have been
        accepted, so the caller must reconcile before doing anything else.
        """
        retries = 0
        last_result = None

        while retries <= self.max_retries:
            result = self.adapter.send_order(request)
            last_result = result

            if result.success:
                logger.info(f"Order executed successfully: deal={result.deal} retcode={result.retcode}")
                return result

            if result.is_requote:
                retries += 1
                logger.warning(f"Order requote/price changed (retcode={result.retcode}). Retry {retries}/{self.max_retries}")
                time.sleep(self.retry_delay_ms / 1000.0)
                continue

            if result.retcode in (RetCode.TIMEOUT, RetCode.CONNECTION):
                # Outcome unknown: the broker may have acted. Never retry.
                logger.error(
                    "Order outcome UNKNOWN (retcode=%s). Not retrying — "
                    "reconcile broker state before resubmitting.",
                    result.retcode,
                )
                return result

            # A definite rejection: the broker did not accept the order.
            logger.error(f"Order rejected (retcode={result.retcode}): {result.comment}")
            break

        return last_result

    # ── Safe path: idempotent submission with explicit execution states ──────

    def submit(
        self,
        request: OrderRequest,
        idempotency_key: str,
    ) -> ExecutionReport:
        """
        Submit an order at most once, returning an explicit execution state.

        Guarantees:

        * A repeated ``idempotency_key`` is never sent again; the original
          report is returned instead.
        * An indeterminate broker outcome (timeout / connection loss) is
          reported as ``UNKNOWN`` and is **never** retried automatically.
          A timeout does not prove the order failed.
        * Only explicit requotes are retried, bounded by ``max_retries``.
        """
        if not self.ledger.begin(idempotency_key):
            existing = self.ledger.get(idempotency_key)
            assert existing is not None  # begin() returned False only if present
            if existing.is_indeterminate:
                logger.error(
                    "Duplicate suppressed for key=%s and prior outcome is UNKNOWN; "
                    "reconcile with the broker.",
                    idempotency_key,
                )
            return existing

        attempts = 0
        report: Optional[ExecutionReport] = None

        while attempts <= self.max_retries:
            attempts += 1
            result = self.adapter.send_order(request)

            state = state_from_retcode(
                result.retcode,
                volume_filled=result.volume,
                volume_requested=request.volume,
            )
            report = ExecutionReport(
                state=state,
                idempotency_key=idempotency_key,
                deal=result.deal,
                order=result.order,
                volume=result.volume,
                price=result.price,
                retcode=result.retcode,
                comment=result.comment,
                attempts=attempts,
            )

            if state in FILLED_STATES or state in (ExecutionState.REJECTED, ExecutionState.CANCELLED):
                break

            if state is ExecutionState.UNKNOWN:
                # Ambiguous. Stop immediately — resubmitting risks a duplicate.
                logger.error(
                    "Submission %s returned UNKNOWN (retcode=%s) after %d attempt(s); "
                    "not retrying.",
                    idempotency_key, result.retcode, attempts,
                )
                break

            # NOT_SENT (requote family): the broker refused this attempt, so a
            # bounded retry with the same key is safe.
            if attempts <= self.max_retries:
                logger.warning(
                    "Submission %s not accepted (retcode=%s); retry %d/%d",
                    idempotency_key, result.retcode, attempts, self.max_retries,
                )
                time.sleep(self.retry_delay_ms / 1000.0)
                continue
            break

        assert report is not None
        self.ledger.complete(report)
        return report

    def reconcile_unresolved_keys(self, magic: int = 888999) -> list[str]:
        """
        Reconcile indeterminate orders (TIMEOUT, CONNECTION) with the broker's actual open positions.
        If a position matching the magic number and ticket/comment is found, updates ledger report.
        Returns the list of remaining unresolved keys.
        """
        unresolved = self.ledger.unresolved_keys()
        if not unresolved:
            return []

        try:
            positions = self.adapter.get_positions(magic=magic)
        except Exception as exc:
            logger.error("Failed to fetch positions during reconciliation: %s", exc)
            return unresolved

        resolved_keys = []
        for key in unresolved:
            record = self.ledger.get(key)
            if not record:
                continue

            # Look for matching position by ticket or comment
            matching_pos = None
            for pos in positions:
                if record.deal and pos.ticket == record.deal:
                    matching_pos = pos
                    break
                if record.order and pos.ticket == record.order:
                    matching_pos = pos
                    break
                if key in pos.comment:
                    matching_pos = pos
                    break

            if matching_pos:
                logger.info("Reconciliation resolved key=%s -> FILLED (ticket=%d)", key, matching_pos.ticket)
                updated_report = ExecutionReport(
                    state=ExecutionState.FILLED,
                    idempotency_key=key,
                    deal=matching_pos.ticket,
                    order=matching_pos.ticket,
                    volume=matching_pos.volume,
                    price=matching_pos.price_open,
                    retcode=RetCode.DONE,
                    comment="Reconciled from broker",
                    attempts=record.attempts,
                )
                self.ledger.complete(updated_report)
                resolved_keys.append(key)

        return [k for k in unresolved if k not in resolved_keys]

