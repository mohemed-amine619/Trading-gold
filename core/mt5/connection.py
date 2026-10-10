"""
core/mt5/connection.py — MT5 connection manager with reconnect logic.

The ConnectionManager is the only object that owns the adapter reference.
It monitors the connection health and attempts reconnect with exponential backoff.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Callable, Optional

from core.mt5.adapter import MT5Adapter
from core.mt5.types import AccountInfo

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class ConnectionManager:
    """
    Manages the MT5 adapter lifecycle.

    Features:
    - Initial connection with configurable timeout
    - Periodic health check in a background thread
    - Exponential backoff on reconnect (5s → 10s → 20s → … → 120s cap)
    - Callbacks for state changes (used by the notification system)
    - Thread-safe state reads

    Usage:
        manager = ConnectionManager(adapter)
        if manager.connect():
            acc = manager.get_account_info()
    """

    BACKOFF_INITIAL   = 5
    BACKOFF_MAX       = 120
    BACKOFF_FACTOR    = 2
    HEALTH_CHECK_INTERVAL = 30   # seconds between background health checks

    def __init__(
        self,
        adapter: MT5Adapter,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
    ) -> None:
        self._adapter = adapter
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_reconnected = on_reconnected

        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._reconnect_backoff = self.BACKOFF_INITIAL

        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog = threading.Event()

    # ── Public API ───────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Attempt initial connection.  Returns True on success."""
        self._set_state(ConnectionState.CONNECTING)
        ok = self._try_connect()
        if ok:
            self._set_state(ConnectionState.CONNECTED)
            self._reconnect_backoff = self.BACKOFF_INITIAL
            self._start_watchdog()
            if self._on_connected:
                self._on_connected()
        else:
            self._set_state(ConnectionState.FAILED)
        return ok

    def disconnect(self) -> None:
        """Gracefully shut down the connection and stop the watchdog."""
        self._stop_watchdog.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5)
        self._adapter.disconnect()
        self._set_state(ConnectionState.DISCONNECTED)

    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

    def get_account_info(self) -> Optional[AccountInfo]:
        """Return account info if connected, else None."""
        if not self.is_connected:
            return None
        return self._adapter.get_account_info()

    def adapter(self) -> MT5Adapter:
        """Direct access to the adapter (for data fetching)."""
        return self._adapter

    # ── Internal ─────────────────────────────────────────────────────────────

    def _try_connect(self) -> bool:
        try:
            return self._adapter.connect()
        except Exception as e:
            logger.error("Connection attempt raised: %s", e)
            return False

    def _set_state(self, state: ConnectionState) -> None:
        with self._state_lock:
            if self._state != state:
                logger.info("Connection state: %s → %s", self._state.value, state.value)
                self._state = state

    def _start_watchdog(self) -> None:
        """Start the background health-check thread."""
        self._stop_watchdog.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="mt5-watchdog"
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.wait(timeout=self.HEALTH_CHECK_INTERVAL):
            if not self._adapter.is_connected():
                logger.warning("MT5 connection lost — attempting reconnect…")
                self._set_state(ConnectionState.RECONNECTING)
                if self._on_disconnected:
                    self._on_disconnected()
                self._reconnect_loop()

    def _reconnect_loop(self) -> None:
        while not self._stop_watchdog.is_set():
            self._adapter.disconnect()
            ok = self._try_connect()
            if ok:
                self._set_state(ConnectionState.CONNECTED)
                self._reconnect_backoff = self.BACKOFF_INITIAL
                logger.info("MT5 reconnected successfully.")
                if self._on_reconnected:
                    self._on_reconnected()
                return

            logger.warning(
                "Reconnect failed — retrying in %ds…", self._reconnect_backoff
            )
            self._stop_watchdog.wait(timeout=self._reconnect_backoff)
            self._reconnect_backoff = min(
                self._reconnect_backoff * self.BACKOFF_FACTOR, self.BACKOFF_MAX
            )

        self._set_state(ConnectionState.FAILED)
        logger.error("Reconnect abandoned — watchdog was stopped.")
