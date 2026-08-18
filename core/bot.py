"""
TradingBot – orchestrator tying the engine, data, strategy, risk and execution
layers together.  Used by BOTH the CLI entry point (main.py) and the PyQt5
GUI (gui_main.py).

New in v2 (AI edition)
-----------------------
  • H4 data fetching       – third timeframe passed to strategy.evaluate()
  • Daily drawdown guard   – skips trading when daily loss limit is breached
  • Breakeven management   – moves SL to entry after 1 ATR profit
  • Partial-close management – closes 50 % of position at 1.5 ATR profit
  • ADX-aware risk sizing  – passes current ADX to RiskManager for scaling

Threading model in GUI mode
---------------------------
The bot owns an asyncio event loop that runs inside a QThread. The GUI never
calls into MT5 or the bot directly:
  * bot → GUI : callbacks (account/positions snapshot, signal feed)
  * GUI → bot : coroutines enqueued with asyncio.run_coroutine_threadsafe()
                (stop, close symbol, close all)
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Set

import MetaTrader5 as mt5

from core.alerts import AlertManager
from core.data_handler import DataHandler
from core.execution import ExecutionEngine
from core.mt5_engine import MT5Engine
from core.risk_manager import RiskManager
from core.state_manager import StateManager
from core.strategy import EmaRsiMultiTimeframeStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback container
# ---------------------------------------------------------------------------

class Callbacks:
    """Hooks fired by the bot; the GUI binds Qt signals to these."""

    def __init__(
        self,
        on_snapshot: Callable[[Optional[dict], List[dict]], None] = None,
        on_signal:   Callable[[dict], None] = None,
    ):
        self.on_snapshot = on_snapshot or (lambda account, positions: None)
        self.on_signal   = on_signal   or (lambda signal: None)


# ---------------------------------------------------------------------------
# TradingBot
# ---------------------------------------------------------------------------

class TradingBot:
    def __init__(self, cfg, callbacks: Optional[Callbacks] = None):
        self.config    = cfg
        self.callbacks = callbacks or Callbacks()
        self.engine    = MT5Engine(cfg)
        self.data      = DataHandler(self.engine, cfg)
        self.risk      = RiskManager(self.engine, cfg)
        self.executor  = ExecutionEngine(self.engine, cfg, self.risk)
        self.state     = StateManager(self.engine, cfg)
        self.alerts    = AlertManager(cfg)
        self.strategies: Dict[str, EmaRsiMultiTimeframeStrategy] = {
            symbol: EmaRsiMultiTimeframeStrategy(symbol, cfg)
            for symbol in cfg.SYMBOLS
        }
        self._entry_bar_times: Dict[str, Optional[object]] = {}
        self._stop_event  = asyncio.Event()
        self.commands: asyncio.Queue = asyncio.Queue()
        self._last_account: Optional[dict] = None

        # Partial-close tracking: set of position tickets already reduced.
        self._partial_closed: Set[int] = set()

    # ------------------------------------------------------------------
    # Control surface (safe from any thread via run_coroutine_threadsafe)
    # ------------------------------------------------------------------
    async def request_stop(self) -> None:
        """Ask the poll loop to exit after the current cycle."""
        self._stop_event.set()

    async def close_symbol(self, symbol: str) -> bool:
        """Market-close the open position on `symbol` (ours only)."""
        pos = self.state.get(symbol)
        if pos is None:
            logger.info("Close requested for %s but no open position", symbol)
            return False
        result = await self.executor.close_position(pos)
        if result.success:
            await self.alerts.trade_exit(symbol, pos.ticket, pos.profit)
            await self.state.refresh()
            self._partial_closed.discard(pos.ticket)
            return True
        return False

    async def close_all(self) -> None:
        """Emergency: market-close every open position we own."""
        for symbol in self.state.symbols():
            await self.close_symbol(symbol)

    async def _process_commands(self) -> None:
        while not self.commands.empty():
            cmd = self.commands.get_nowait()
            if cmd[0] == "close":
                await self.close_symbol(cmd[1])
            elif cmd[0] == "close_all":
                await self.close_all()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def startup(self) -> None:
        await self.engine.connect()
        for symbol in self.config.SYMBOLS:
            await self.engine.activate_symbol(symbol)
            self._entry_bar_times[symbol] = await self.data.last_closed_bar_time(
                symbol, self.config.ENTRY_TIMEFRAME)
        account = await self.engine.call(mt5.account_info)
        # Seed the daily tracker so the first poll cycle has a valid baseline.
        if account:
            self.risk.daily_tracker.update(account.equity)
        await self.alerts.startup(account.server, account.login, account.balance)

    async def shutdown(self) -> None:
        await self.engine.disconnect()
        await self.alerts.shutdown()

    async def run_until_stopped(self) -> None:
        """Run the poll loop until request_stop() is called (or the task is cancelled)."""
        await self.startup()
        logger.info("Bot started. Polling every %.1fs", self.config.POLL_INTERVAL_SECONDS)
        while not self._stop_event.is_set():
            try:
                await self._process_commands()
                await self.tick()
            except Exception as exc:
                logger.exception("Unexpected error in tick()")
                await self.alerts.crash(repr(exc))
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
        logger.info("Poll loop exited (stop requested)")

    # ------------------------------------------------------------------
    # One poll cycle
    # ------------------------------------------------------------------
    async def tick(self) -> None:
        if not await self.engine.ensure_connected():
            return

        positions = await self.state.refresh()
        account   = await self.engine.call(mt5.account_info)
        self._last_account = None if account is None else {
            "server":      account.server,    "login":       account.login,
            "balance":     account.balance,   "equity":      account.equity,
            "margin":      account.margin,    "margin_free": account.margin_free,
            "profit":      account.profit,    "currency":    account.currency,
        }
        equity = account.equity if account else 0.0

        # --- daily drawdown guard ----------------------------------------
        if account:
            self.risk.daily_tracker.update(equity)
            max_dd = getattr(self.config, "MAX_DAILY_DRAWDOWN_PCT", 0.03)
            if self.risk.daily_tracker.is_limit_hit(equity, max_dd):
                logger.warning("Trading halted for today: daily drawdown limit reached")
                self._emit_snapshot()
                return

        # --- per-symbol loop ---------------------------------------------
        for symbol in self.config.SYMBOLS:
            strategy = self.strategies[symbol]

            bar_time = await self.data.last_closed_bar_time(
                symbol, self.config.ENTRY_TIMEFRAME)
            if bar_time is None:
                continue

            new_bar = (bar_time != self._entry_bar_times.get(symbol))
            if new_bar:
                self._entry_bar_times[symbol] = bar_time
                info = await self.data.get_symbol_info(symbol)

                # --- fetch all three timeframes --------------------------
                entry_df  = await self.data.fetch_candles(
                    symbol, self.config.ENTRY_TIMEFRAME, self.config.HISTORY_BARS)
                filter_df = await self.data.fetch_candles(
                    symbol, self.config.FILTER_TIMEFRAME, self.config.HISTORY_BARS)
                h4_df     = await self.data.fetch_candles(
                    symbol,
                    getattr(self.config, "H4_TIMEFRAME", "H4"),
                    getattr(self.config, "H4_HISTORY_BARS", 300))

                if entry_df.empty or filter_df.empty:
                    logger.warning("Insufficient data for %s", symbol)
                    continue

                # --- prepare (attach indicators) -------------------------
                entry_df  = strategy.prepare(entry_df)
                filter_df = strategy.prepare(filter_df)
                h4_df     = (strategy.prepare(h4_df)
                             if not h4_df.empty else None)

                # --- evaluate --------------------------------------------
                signal = strategy.evaluate(entry_df, filter_df, h4_df)
                self._emit_signal(symbol, bar_time, signal)

                # --- position management (trailing + breakeven + partial)-
                await self._manage_trailing(symbol, strategy, positions)
                await self._manage_breakeven(symbol, strategy, positions)
                await self._manage_partial_close(symbol, strategy, positions, info)

                # --- entry logic -----------------------------------------
                if signal.direction == 0:
                    continue
                if self.state.has_position(symbol):
                    logger.debug("%s: signal %+d ignored – position already open",
                                 symbol, signal.direction)
                    continue
                if self.state.count() >= self.config.MAX_OPEN_POSITIONS:
                    logger.info("%s: signal %+d ignored – max positions reached",
                                symbol, signal.direction)
                    continue

                # --- ADX for dynamic risk sizing -------------------------
                adx = 0.0
                if "adx" in entry_df.columns:
                    adx = float(entry_df.iloc[-1].get("adx", 0.0))

                # --- risk sizing + margin check --------------------------
                sizing = await self.risk.size_position(
                    info, signal.direction, signal.close,
                    signal.atr, equity, adx=adx)
                if sizing.rejected:
                    logger.warning("%s: trade rejected – %s", symbol, sizing.reason)
                    continue

                # --- order execution -------------------------------------
                result = await self.executor.market_order(
                    info, signal.direction, sizing.volume, sizing.sl, sizing.tp)
                if result.success:
                    await self.alerts.trade_entry(
                        symbol, signal.direction, sizing.volume,
                        signal.close, sizing.sl, sizing.tp)
                    await self.state.refresh()

            else:
                # No new candle: still manage open positions every poll cycle.
                await self._manage_trailing(symbol, strategy, positions)
                await self._manage_breakeven(symbol, strategy, positions)
                # Note: partial close is only checked on new bar to avoid
                # repeated triggers on the same position.

        self._emit_snapshot()

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------
    async def _manage_trailing(self, symbol: str, strategy, positions) -> None:
        """Ratchet trailing stop on the most recent closed bar's indicators."""
        pos = positions.get(symbol)
        if pos is None or pos.sl is None:
            return
        snapshot = strategy.snapshot()
        if snapshot.atr <= 0:
            return
        new_sl = self.risk.trailing_stop(
            pos.direction, snapshot.close, snapshot.atr, pos.entry, pos.sl)
        if new_sl is not None and abs(new_sl - pos.sl) > 1e-9:
            result = await self.executor.modify_sl_tp(symbol, pos.ticket, new_sl, pos.tp)
            if result.success:
                pos.sl = new_sl
                await self.alerts.trailing_update(symbol, pos.ticket, new_sl)

    async def _manage_breakeven(self, symbol: str, strategy, positions) -> None:
        """Move SL to breakeven once the position has profited enough."""
        pos = positions.get(symbol)
        if pos is None:
            return
        snapshot = strategy.snapshot()
        if snapshot.atr <= 0 or snapshot.close == 0.0:
            return
        be = self.risk.check_breakeven(
            pos.direction, pos.entry, pos.sl, snapshot.close, snapshot.atr)
        if not be.should_move:
            return
        result = await self.executor.modify_sl_tp(symbol, pos.ticket, be.new_sl, pos.tp)
        if result.success:
            pos.sl = be.new_sl
            logger.info("BREAKEVEN %s ticket=%s → sl=entry=%.5f",
                        symbol, pos.ticket, be.new_sl)
            await self.alerts.trailing_update(symbol, pos.ticket, be.new_sl)

    async def _manage_partial_close(self, symbol: str, strategy, positions,
                                    info) -> None:
        """Close half the position once it reaches 1.5 ATR in profit."""
        pos = positions.get(symbol)
        if pos is None or pos.ticket in self._partial_closed:
            return
        snapshot = strategy.snapshot()
        if snapshot.atr <= 0 or snapshot.close == 0.0:
            return
        pc = self.risk.check_partial_close(
            pos.direction, pos.entry, pos.volume,
            snapshot.close, snapshot.atr,
            info.volume_step, info.volume_min)
        if not pc.should_close:
            return
        result = await self.executor.partial_close_position(pos, pc.volume)
        if result.success:
            self._partial_closed.add(pos.ticket)
            logger.info("PARTIAL CLOSE %s ticket=%s vol=%.2f",
                        symbol, pos.ticket, pc.volume)
            # Refresh state to reflect the reduced position.
            positions = await self.state.refresh()

    # ------------------------------------------------------------------
    # Telemetry callbacks (consumed by the GUI)
    # ------------------------------------------------------------------
    def _emit_snapshot(self) -> None:
        pos_list = [{
            "ticket": p.ticket, "symbol": p.symbol,
            "direction": p.direction, "volume": p.volume,
            "entry": p.entry, "sl": p.sl, "tp": p.tp, "profit": p.profit,
        } for p in self.state.open_positions()]
        self.callbacks.on_snapshot(self._last_account, pos_list)

    def _emit_signal(self, symbol: str, bar_time, signal) -> None:
        if signal.direction == 0:
            return
        self.callbacks.on_signal({
            "time":         str(bar_time),
            "symbol":       symbol,
            "direction":    signal.direction,
            "reason":       signal.reason,
            "close":        signal.close,
            "score":        getattr(signal, "score", 0.0),
            "ai_confidence": getattr(signal, "ai_confidence", 0.0),
        })
