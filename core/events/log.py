"""
core/events/log.py — Immutable structured event logger.

Every meaningful action in the system produces an event:
  - Signal generated
  - AI decision
  - Risk approved / rejected
  - Order submitted / filled / rejected
  - Position opened / closed / modified
  - Risk limits hit
  - Manual actions
  - System start / stop

Events are written to:
  1. The database (SystemEvent table) for permanent storage
  2. The Python structlog logger for real-time visibility
  3. A broadcast queue for WebSocket live feed
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global broadcast queue (WebSocket layer subscribes to this)
_broadcast_queue: queue.Queue = queue.Queue(maxsize=1000)

# Optional database session factory (injected at startup)
_db_session_factory = None


def configure_event_log(db_session_factory) -> None:
    """
    Call once at startup to enable database persistence.
    Pass a callable that returns a context-manager yielding a sync session.
    """
    global _db_session_factory
    _db_session_factory = db_session_factory


def get_broadcast_queue() -> queue.Queue:
    """Return the queue that WebSocket broadcaster reads from."""
    return _broadcast_queue


class EventLog:
    """
    Thread-safe event logger.

    Usage:
        EventLog.emit(
            event_type="POSITION_CLOSED",
            message="EURUSD BUY closed | pnl=+$72.40",
            symbol="EURUSD",
            position_id="abc-123",
            details={"pnl": 72.40, "reason": "trailing_stop"},
        )
    """

    _lock = threading.Lock()

    @classmethod
    def emit(
        cls,
        event_type: str,
        message: str,
        symbol: Optional[str] = None,
        position_id: Optional[str] = None,
        order_id: Optional[str] = None,
        mt5_ticket: Optional[int] = None,
        details: Optional[dict] = None,
        persist: bool = True,
    ) -> None:
        """
        Emit an event to all sinks (log, DB, broadcast queue).

        Args:
            event_type:   EventType enum value (string).
            message:      Human-readable description.
            symbol:       Trading symbol (optional).
            position_id:  Internal position UUID (optional).
            order_id:     Internal order UUID (optional).
            mt5_ticket:   MT5 ticket number (optional).
            details:      Arbitrary key-value context dict (optional).
            persist:      If False, skip database write (e.g. high-frequency heartbeats).
        """
        now = datetime.now(timezone.utc)
        details_json = json.dumps(details) if details else None

        # 1. Python logger
        log_msg = f"[EVENT:{event_type}]"
        if symbol:
            log_msg += f" {symbol}"
        log_msg += f" | {message}"
        logger.info(log_msg, extra={"details": details or {}})

        # 2. Broadcast queue (non-blocking — drop if full)
        event_payload = {
            "type": event_type,
            "message": message,
            "symbol": symbol,
            "position_id": position_id,
            "order_id": order_id,
            "mt5_ticket": mt5_ticket,
            "details": details,
            "timestamp": now.isoformat(),
        }
        try:
            _broadcast_queue.put_nowait(event_payload)
        except queue.Full:
            pass  # WebSocket consumer fell behind; skip this event rather than blocking

        # 3. Database persistence (in a background thread to avoid blocking the loop)
        if persist and _db_session_factory is not None:
            snapshot = {
                "event_type": event_type,
                "message": message,
                "symbol": symbol,
                "position_id": position_id,
                "order_id": order_id,
                "mt5_ticket": mt5_ticket,
                "details_json": details_json,
                "created_at": now,
            }
            threading.Thread(
                target=cls._persist,
                args=(snapshot,),
                daemon=True,
                name="event-persist",
            ).start()

    @classmethod
    def _persist(cls, snapshot: dict) -> None:
        try:
            with _db_session_factory() as session:
                from core.database.models import SystemEvent
                event = SystemEvent(
                    event_type=snapshot["event_type"],
                    symbol=snapshot["symbol"],
                    position_id=snapshot["position_id"],
                    order_id=snapshot["order_id"],
                    mt5_ticket=snapshot["mt5_ticket"],
                    message=snapshot["message"],
                    details=snapshot["details_json"],
                    created_at=snapshot["created_at"],
                )
                session.add(event)
                # session auto-committed by context manager
        except Exception as exc:
            logger.error("EventLog: failed to persist event: %s", exc)

    # ── Convenience helpers ────────────────────────────────────────────────

    @classmethod
    def signal(cls, symbol: str, direction: int, score: float, reason: str, **kw) -> None:
        cls.emit(
            "SIGNAL_GENERATED",
            f"{'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'NONE'} "
            f"score={score:.1f} | {reason}",
            symbol=symbol,
            details={"direction": direction, "score": score, "reason": reason, **kw},
        )

    @classmethod
    def risk_rejected(cls, symbol: str, reason: str, **kw) -> None:
        cls.emit(
            "RISK_REJECTED",
            f"Trade rejected: {reason}",
            symbol=symbol,
            details={"reason": reason, **kw},
        )

    @classmethod
    def risk_approved(cls, symbol: str, lots: float, sl: float, tp: float, **kw) -> None:
        cls.emit(
            "RISK_APPROVED",
            f"Trade approved: {lots:.2f} lots | SL={sl:.5f} TP={tp:.5f}",
            symbol=symbol,
            details={"lots": lots, "sl": sl, "tp": tp, **kw},
        )

    @classmethod
    def position_opened(cls, symbol: str, ticket: int, side: str, lots: float,
                        price: float, sl: float, tp: float, **kw) -> None:
        cls.emit(
            "POSITION_OPENED",
            f"{side} {symbol} | ticket={ticket} | {lots:.2f}L @ {price:.5f} "
            f"SL={sl:.5f} TP={tp:.5f}",
            symbol=symbol,
            mt5_ticket=ticket,
            details={"side": side, "lots": lots, "price": price, "sl": sl, "tp": tp, **kw},
        )

    @classmethod
    def position_closed(cls, symbol: str, ticket: int, pnl: float, reason: str, **kw) -> None:
        cls.emit(
            "POSITION_CLOSED",
            f"CLOSED {symbol} | ticket={ticket} | pnl={pnl:+.2f} | reason={reason}",
            symbol=symbol,
            mt5_ticket=ticket,
            details={"pnl": pnl, "reason": reason, **kw},
        )

    @classmethod
    def stop_moved(cls, symbol: str, ticket: int, old_sl: float, new_sl: float, **kw) -> None:
        cls.emit(
            "STOP_MOVED",
            f"SL moved: {old_sl:.5f} → {new_sl:.5f}",
            symbol=symbol,
            mt5_ticket=ticket,
            details={"old_sl": old_sl, "new_sl": new_sl, **kw},
            persist=False,  # high-frequency; skip DB for performance
        )

    @classmethod
    def daily_limit_hit(cls, daily_dd_pct: float, equity: float) -> None:
        cls.emit(
            "DAILY_LIMIT_HIT",
            f"Daily drawdown limit hit: {daily_dd_pct:.2%} | equity={equity:.2f}",
            details={"daily_dd_pct": daily_dd_pct, "equity": equity},
        )

    @classmethod
    def emergency_stop(cls, reason: str, triggered_by: str = "system") -> None:
        cls.emit(
            "EMERGENCY_STOP",
            f"EMERGENCY STOP triggered by {triggered_by}: {reason}",
            details={"reason": reason, "triggered_by": triggered_by},
        )

    @classmethod
    def manual_action(cls, action: str, performed_by: str, **kw) -> None:
        cls.emit(
            "MANUAL_ACTION",
            f"Manual: {action} by {performed_by}",
            details={"action": action, "performed_by": performed_by, **kw},
        )

    @classmethod
    def mt5_disconnected(cls) -> None:
        cls.emit("MT5_DISCONNECTED", "MT5 connection lost")

    @classmethod
    def mt5_reconnected(cls) -> None:
        cls.emit("MT5_RECONNECTED", "MT5 connection restored")

    @classmethod
    def ai_unavailable(cls, provider: str, error: str) -> None:
        cls.emit(
            "AI_UNAVAILABLE",
            f"LLM provider '{provider}' unavailable: {error}",
            details={"provider": provider, "error": error},
        )

    @classmethod
    def heartbeat(cls, status: dict) -> None:
        cls.emit("HEARTBEAT", "System heartbeat", details=status, persist=False)
