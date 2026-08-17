"""
Trading bot - entry point & async main event loop.

Architecture
------------
main.py ................... wiring + async event loop
config.py ................. every tunable parameter
core/mt5_engine.py ......... MT5 lifecycle + serialized blocking calls
core/data_handler.py ....... OHLCV fetching + timezone normalization
core/strategy.py ........... signal generation on closed candles
core/risk_manager.py ....... sizing, SL/TP, margin checks, ATR trailing stop
core/execution.py .......... order_send wrappers + retcode handling
core/state_manager.py ...... mirror of our open positions
core/alerts.py ............. Telegram / Discord notifications

Loop timing
-----------
The loop wakes every POLL_INTERVAL_SECONDS and:
  1. verifies the terminal connection (reconnects if needed)
  2. refreshes the position mirror / account equity
  3. checks whether a NEW candle closed on the entry timeframe
  4. only then re-computes indicators & signals (cheap, and impossible to
     repaint because forming candles are dropped by the DataHandler)
  5. applies the ATR trailing stop to open positions
  6. sends a market order if the signal fires and all risk checks pass

All MT5 calls go through the engine's single worker thread, so the asyncio
loop never blocks on the terminal.

Graceful shutdown
-----------------
Ctrl+C raises KeyboardInterrupt inside asyncio.run() -> the main task is
cancelled, mt5.shutdown() runs, an alert is fired, and no order is left
half-processed.
"""
import asyncio
import logging
import logging.handlers
import os
from typing import Dict, Optional

import MetaTrader5 as mt5

import config
from core.alerts import AlertManager
from core.data_handler import DataHandler
from core.execution import ExecutionEngine
from core.mt5_engine import MT5Engine
from core.risk_manager import RiskManager
from core.state_manager import StateManager
from core.strategy import EmaRsiMultiTimeframeStrategy

logger = logging.getLogger(__name__)


def setup_logging(cfg) -> None:
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(cfg.LOG_DIR, cfg.LOG_FILE),
        maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Keep the MetaTrader5 package's own (very noisy) logs quiet.
    logging.getLogger("MetaTrader5").setLevel(logging.ERROR)


class TradingBot:
    def __init__(self, cfg):
        self.config = cfg
        self.engine = MT5Engine(cfg)
        self.data = DataHandler(self.engine, cfg)
        self.risk = RiskManager(self.engine, cfg)
        self.executor = ExecutionEngine(self.engine, cfg, self.risk)
        self.state = StateManager(self.engine, cfg)
        self.alerts = AlertManager(cfg)
        self.strategies = {
            symbol: EmaRsiMultiTimeframeStrategy(symbol, cfg)
            for symbol in cfg.SYMBOLS
        }
        self._entry_bar_times: Dict[str, Optional[object]] = {}
        self._running = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def startup(self) -> None:
        await self.engine.connect()
        for symbol in self.config.SYMBOLS:
            # Symbols must be enabled in MarketWatch before they can be traded.
            await self.engine.activate_symbol(symbol)
            self._entry_bar_times[symbol] = await self.data.last_closed_bar_time(
                symbol, self.config.ENTRY_TIMEFRAME)
        account = await self.engine.call(mt5.account_info)
        await self.alerts.startup(account.server, account.login, account.balance)

    async def run(self) -> None:
        await self.startup()
        self._running = True
        logger.info("Bot started. Polling every %.1fs", self.config.POLL_INTERVAL_SECONDS)
        while self._running:
            try:
                await self.tick()
            except Exception as exc:
                # Never die silently: log, alert, and keep polling.
                logger.exception("Unexpected error in tick()")
                await self.alerts.crash(repr(exc))
            await asyncio.sleep(self.config.POLL_INTERVAL_SECONDS)

    # ------------------------------------------------------------------
    # one poll cycle
    # ------------------------------------------------------------------
    async def tick(self) -> None:
        if not await self.engine.ensure_connected():
            return
        positions = await self.state.refresh()
        equity = await self.state.equity()

        for symbol in self.config.SYMBOLS:
            strategy = self.strategies[symbol]

            # --- new closed bar on the entry timeframe? -------------------
            bar_time = await self.data.last_closed_bar_time(
                symbol, self.config.ENTRY_TIMEFRAME)
            if bar_time is None:
                continue

            if bar_time != self._entry_bar_times.get(symbol):
                self._entry_bar_times[symbol] = bar_time
                info = await self.data.get_symbol_info(symbol)
                entry_df = await self.data.fetch_candles(
                    symbol, self.config.ENTRY_TIMEFRAME, self.config.HISTORY_BARS)
                filter_df = await self.data.fetch_candles(
                    symbol, self.config.FILTER_TIMEFRAME, self.config.HISTORY_BARS)
                if entry_df.empty or filter_df.empty:
                    logger.warning("Insufficient data for %s", symbol)
                    continue

                # Vectorized indicators, evaluated on closed bars only.
                entry_df = strategy.prepare(entry_df)
                filter_df = strategy.prepare(filter_df)
                signal = strategy.evaluate(entry_df, filter_df)

                # --- trailing stop upkeep on the new bar ------------------
                await self._manage_trailing(symbol, strategy, positions)

                # --- entry logic ------------------------------------------
                if signal.direction == 0:
                    continue
                if self.state.has_position(symbol):
                    logger.debug("%s: signal %+d ignored, position already open",
                                 symbol, signal.direction)
                    continue
                if self.state.count() >= self.config.MAX_OPEN_POSITIONS:
                    logger.info("%s: signal %+d ignored, max positions reached",
                                symbol, signal.direction)
                    continue

                # --- risk checks + sizing (margin verified inside) --------
                sizing = await self.risk.size_position(
                    info, signal.direction, signal.close, signal.atr, equity)
                if sizing.rejected:
                    logger.warning("%s: trade rejected - %s", symbol, sizing.reason)
                    continue

                # --- execution ---------------------------------------------
                result = await self.executor.market_order(
                    info, signal.direction, sizing.volume, sizing.sl, sizing.tp)
                if result.success:
                    await self.alerts.trade_entry(
                        symbol, signal.direction, sizing.volume,
                        signal.close, sizing.sl, sizing.tp)
                    await self.state.refresh()   # mirror the new position
            else:
                # No new candle: only trailing-stop upkeep is possible.
                await self._manage_trailing(symbol, strategy, positions)

    # ------------------------------------------------------------------
    # trailing stop
    # ------------------------------------------------------------------
    async def _manage_trailing(self, symbol: str, strategy, positions) -> None:
        pos = positions.get(symbol)
        if pos is None or pos.sl is None:
            return
        snapshot = strategy.snapshot()   # close + ATR of the last CLOSED bar
        if snapshot.atr <= 0:
            return
        new_sl = self.risk.trailing_stop(
            pos.direction, snapshot.close, snapshot.atr, pos.entry, pos.sl)
        if new_sl is not None and abs(new_sl - pos.sl) > 1e-9:
            result = await self.executor.modify_sl_tp(symbol, pos.ticket, new_sl, pos.tp)
            if result.success:
                pos.sl = new_sl   # update the mirror in place
                await self.alerts.trailing_update(symbol, pos.ticket, new_sl)


async def main() -> None:
    bot = TradingBot(config)
    try:
        await bot.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested (Ctrl+C)")
    except Exception as exc:
        logger.exception("Fatal error in main loop")
        await bot.alerts.crash(repr(exc))
    finally:
        bot._running = False
        await bot.engine.disconnect()
        await bot.alerts.shutdown()
        logger.info("Bot stopped cleanly")


if __name__ == "__main__":
    setup_logging(config)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
