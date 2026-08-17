"""
MT5Engine - thin, defensive wrapper around the official MetaTrader5 package.

Why a dedicated wrapper?
  * The MT5 python bindings are NOT thread-safe and their calls block until the
    terminal answers. All calls are therefore dispatched to a single worker
    thread (ThreadPoolExecutor(max_workers=1)) so the asyncio loop never
    stalls and no two MT5 calls can interleave.
  * The wrapper centralizes initialize/login/shutdown lifecycle handling and
    transparent reconnection when the terminal link drops.

Error convention: `mt5.last_error()` returns a (code, description) tuple where
code == 0 means success. Every wrapper raises MT5Error on hard failures so the
caller can decide to retry or skip the symbol.
"""
import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class MT5Error(Exception):
    """Raised when an MT5 call fails or returns an empty/invalid result."""


class MT5Engine:
    def __init__(self, config):
        self.config = config
        self._connected = False
        # A single worker keeps every MT5 call serialized; never increase this.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")
        self._loop = None

    # ------------------------------------------------------------------
    # async plumbing
    # ------------------------------------------------------------------
    async def call(self, func, *args, **kwargs):
        """Run a blocking MT5 function in the single worker thread."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return await self._loop.run_in_executor(
            self._executor, functools.partial(func, *args, **kwargs)
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> bool:
        """Initialize the terminal and (optionally) log into the account."""
        # Build kwargs conditionally: mt5.initialize's default login=0 means
        # "use the account already logged into the terminal".
        kwargs = {}
        if self.config.MT5_PATH:
            kwargs["path"] = self.config.MT5_PATH
        if self.config.MT5_LOGIN:
            kwargs["login"] = self.config.MT5_LOGIN
        if self.config.MT5_PASSWORD:
            kwargs["password"] = self.config.MT5_PASSWORD
        if self.config.MT5_SERVER:
            kwargs["server"] = self.config.MT5_SERVER

        if not await self.call(mt5.initialize, **kwargs):
            code, desc = await self.call(mt5.last_error)
            raise MT5Error(f"mt5.initialize failed: {desc} (code {code})")

        # If the terminal opened on a different account than configured, log in
        # explicitly. mt5.login(login, password='', server='') - all positional
        # args defaulted by us.
        if self.config.MT5_LOGIN:
            account = await self.call(mt5.account_info)
            if account is None or account.login != self.config.MT5_LOGIN:
                login_kwargs = {"login": self.config.MT5_LOGIN}
                if self.config.MT5_PASSWORD:
                    login_kwargs["password"] = self.config.MT5_PASSWORD
                if self.config.MT5_SERVER:
                    login_kwargs["server"] = self.config.MT5_SERVER
                if not await self.call(mt5.login, **login_kwargs):
                    code, desc = await self.call(mt5.last_error)
                    raise MT5Error(f"mt5.login failed: {desc} (code {code})")

        account = await self.call(mt5.account_info)
        if account is None:
            raise MT5Error("mt5.account_info returned None after connect")
        self._connected = True
        logger.info(
            "Connected: %s #%s balance=%.2f equity=%.2f margin_free=%.2f",
            account.server, account.login, account.balance, account.equity,
            account.margin_free,
        )
        return True

    async def disconnect(self) -> None:
        """Release the terminal. Safe to call multiple times (shutdown idempotent)."""
        if self._connected:
            try:
                await self.call(mt5.shutdown)
            except Exception:  # never let shutdown mask the real error
                pass
            self._connected = False
            logger.info("MT5 terminal shut down")

    async def ensure_connected(self) -> bool:
        """Return True if the terminal link is alive; attempt reconnection."""
        terminal = await self.call(mt5.terminal_info)
        connected = terminal is not None and terminal.connected
        if connected:
            self._connected = True
            return True
        logger.warning("MT5 terminal disconnected, reconnecting...")
        await self.disconnect()
        await asyncio.sleep(1.0)
        return await self.connect()

    async def activate_symbol(self, symbol: str) -> bool:
        """Enable a symbol in MarketWatch.

        mt5.symbol_select(symbol, True) adds the instrument to MarketWatch,
        which brokers require before the symbol can be priced/traded. Returns
        False if the symbol does not exist on this server.
        """
        ok = await self.call(mt5.symbol_select, symbol, True)
        if not ok:
            code, desc = await self.call(mt5.last_error)
            logger.error("symbol_select(%s) failed: %s (code %s)", symbol, desc, code)
        return ok
