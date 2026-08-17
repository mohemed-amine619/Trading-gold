"""
ExecutionEngine - the ONLY module that calls mt5.order_send.

MT5 error code handling ("magic numbers")
-----------------------------------------
Every order_send returns a result with a numeric `retcode`. The important ones:

    10004 / 10023  REQUOTE ........... price moved while the order was in
                                      flight; re-fetch the LIVE price and
                                      resubmit
    10020          PRICE CHANGED ..... ditto (market flavour)
    10022          TRADE CONTEXT BUSY   the broker engine is still busy with
                                      another request; wait and retry
    10016          INVALID STOPS ..... SL/TP too close to price; widen them to
                                      the symbol's SYMBOL_TRADE_STOPS_LEVEL
    10014          INVALID VOLUME .... volume below min or off the lot step
    10018          MARKET CLOSED ..... session closed; skip until next poll
    10019 / 10031 / 10032  NO MONEY / INCORRECT FUNDS / MARGIN CALLS
    10009          DONE .............. success (result.order = deal id)
    10008          PLACED ............ success (pending orders)

Deviation parameter
-------------------
`deviation` = maximum acceptable slippage for market orders, in POINTS
(symbol_info.point units). MT5 will only fill within this many points of the
requested price, otherwise it returns REQUOTE. 0 disables slippage but causes
constant requotes on fast markets; 20-50 points is a reasonable compromise.

Fill modes
----------
Market orders require an explicit fill policy. The legal modes are advertised
by the broker in SYMBOL_FILLING_MODE as a bitmask:
    ORDER_FILLING_FOK    (1) fill the whole volume at once or cancel
    ORDER_FILLING_IOC    (2) fill whatever is available, cancel the rest
    ORDER_FILLING_RETURN (4) fill partially, remainder becomes a pending order
We pick the most permissive legal mode (RETURN > IOC > FOK), falling back to
FOK for brokers that only allow it (common for CFDs).

Position ticket note: for TRADE_ACTION_DEAL market orders, result.order is the
DEAL ticket. Position tickets may differ on some brokers, so the caller should
reconcile via positions_get() filtered by magic number (StateManager does it).
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

# Human-readable names for the most common TRADE_RETCODE_* values.
RETCODE_NAMES = {
    10004: "Requote", 10006: "Request rejected", 10007: "Request canceled by trader",
    10008: "Order placed", 10009: "Order done", 10010: "Only part of the request completed",
    10011: "Request processing error", 10012: "Request canceled by timeout",
    10013: "Invalid request", 10014: "Invalid volume", 10015: "Invalid price",
    10016: "Invalid stops", 10017: "Trade disabled", 10018: "Market closed",
    10019: "No money", 10020: "Price changed", 10021: "No quotes",
    10022: "Broker busy", 10023: "Requote", 10024: "Position locked",
    10025: "Position closed", 10026: "Too many requests", 10027: "Request confirmed",
    10028: "Too frequent requests", 10029: "No connection", 10030: "Authorization failed",
    10031: "Incorrect funds", 10032: "Margin calls", 10033: "Trade expired",
    10034: "Too many pending orders", 10035: "Invalid order expiration",
    10036: "Too many modifications", 10037: "Order locked", 10038: "Order closed",
    10039: "Order rejected", 10040: "Too many order modifications",
    10041: "Dealer refused", 10042: "Options expired", 10043: "No quotes",
    10044: "Order is not valid", 10045: "Not enough rights", 10046: "Position not found",
    10047: "Unknown symbol", 10048: "Account balance is invalid",
    10049: "Trade time expired", 10050: "No trade activity", 10051: "Account disabled",
    10052: "Duplicate order",
}

DONE = 10009
PLACED = 10008
INVALID_STOPS = 10016
MARKET_CLOSED = 10018
NO_MONEY = {10019, 10031, 10032}
RETRYABLE = {10004, 10020, 10022, 10023}   # requote / price changed / context busy


@dataclass
class OrderResult:
    success: bool
    retcode: int
    retcode_name: str
    ticket: Optional[int] = None
    volume: float = 0.0
    message: str = ""


class ExecutionEngine:
    def __init__(self, engine, config, risk_manager):
        self.engine = engine
        self.config = config
        self.risk = risk_manager

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _filling_mode(self, symbol_info) -> int:
        """Pick the most permissive fill mode the broker allows (see docstring)."""
        mode = symbol_info.filling_mode
        if mode & mt5.SYMBOL_FILLING_RETURN:
            return mt5.ORDER_FILLING_RETURN
        if mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_FOK

    async def _live_price(self, symbol: str, direction: int) -> Optional[float]:
        """Best current price: ask for buys, bid for sells."""
        tick = await self.engine.call(mt5.symbol_info_tick, symbol)
        if tick is None:
            return None
        return tick.ask if direction == 1 else tick.bid

    async def _submit(self, request: dict, symbol_info, direction: int) -> OrderResult:
        """Dispatch an order_send request with retcode-aware retry logic."""
        c = self.config
        for attempt in range(1, c.MAX_REQUOTE_RETRIES + 1):
            result = await self.engine.call(mt5.order_send, request)
            if result is None:
                code, desc = await self.engine.call(mt5.last_error)
                logger.error("order_send returned None: %s (code %s)", desc, code)
                await asyncio.sleep(c.RETRY_BACKOFF_SECONDS)
                continue

            retcode = result.retcode
            name = RETCODE_NAMES.get(retcode, f"unknown({retcode})")

            if retcode in (DONE, PLACED):
                return OrderResult(True, retcode, name, ticket=result.order,
                                   volume=request["volume"], message=name)

            if retcode in RETRYABLE:
                # Price moved / broker busy: refresh the price and resubmit.
                logger.warning("%s on %s (attempt %d/%d), resubmitting at fresh price",
                               name, request["symbol"], attempt, c.MAX_REQUOTE_RETRIES)
                await asyncio.sleep(c.RETRY_BACKOFF_SECONDS)
                new_price = await self._live_price(request["symbol"], direction)
                if new_price is not None:
                    request["price"] = new_price
                    # Re-clamp SL/TP relative to the fresh price.
                    if request.get("sl") is not None and request.get("tp") is not None:
                        sl, tp = self.risk.clamp_stops(
                            new_price, request["sl"], request["tp"], symbol_info)
                        request["sl"], request["tp"] = sl, tp
                continue

            if retcode == INVALID_STOPS:
                # Stops inside the broker's minimum distance: widen them.
                logger.warning("Invalid stops on %s (attempt %d/%d), widening",
                               request["symbol"], attempt, c.MAX_REQUOTE_RETRIES)
                await asyncio.sleep(c.RETRY_BACKOFF_SECONDS)
                min_dist = symbol_info.stops_level * symbol_info.point
                if request.get("sl"):
                    # long: push SL down / short: push SL up (away from price)
                    request["sl"] -= min_dist if direction == 1 else -min_dist
                if request.get("tp"):
                    request["tp"] += min_dist if direction == 1 else -min_dist
                continue

            if retcode in NO_MONEY or retcode == MARKET_CLOSED:
                logger.error("Blocked %s: %s", request["symbol"], name)
                return OrderResult(False, retcode, name, message=name)

            logger.error("order_send failed for %s: %s (retcode %d)",
                         request["symbol"], name, retcode)
            return OrderResult(False, retcode, name, message=name)

        return OrderResult(False, -1, "retries exhausted",
                           message="gave up after requote/context-busy retries")

    # ------------------------------------------------------------------
    # public operations
    # ------------------------------------------------------------------
    async def market_order(self, symbol_info, direction: int, volume: float,
                           sl: float, tp: float) -> OrderResult:
        """Open a market position (buy/sell) with SL/TP, requote-safe."""
        price = await self._live_price(symbol_info.name, direction)
        if price is None:
            return OrderResult(False, -1, "no tick",
                               message=f"symbol_info_tick({symbol_info.name}) failed")
        sl, tp = self.risk.clamp_stops(price, sl, tp, symbol_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,   # immediate market execution
            "symbol": symbol_info.name,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.config.DEVIATION_POINTS,  # max slippage in points
            "magic": self.config.MAGIC_NUMBER,          # our order fingerprint
            "comment": f"bot {direction:+d}",
            "type_time": mt5.ORDER_TIME_GTC,            # Good Till Cancelled
            "type_filling": self._filling_mode(symbol_info),
        }
        result = await self._submit(request, symbol_info, direction)
        if result.success:
            logger.info("OPEN %s %s vol=%.2f ticket=%s sl=%.5f tp=%.5f",
                        "BUY" if direction == 1 else "SELL", symbol_info.name,
                        volume, result.ticket, sl, tp)
        return result

    async def modify_sl_tp(self, symbol: str, ticket: int, sl: float,
                           tp: float) -> OrderResult:
        """Update the SL/TP of an existing position (used by trailing stops)."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,   # modify stops of a position
            "symbol": symbol,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.config.MAGIC_NUMBER,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = await self.engine.call(mt5.order_send, request)
        if result is None or result.retcode != DONE:
            name = RETCODE_NAMES.get(result.retcode, f"unknown({result.retcode})") \
                if result else "no result"
            logger.error("SL/TP modify failed for position %s: %s", ticket, name)
            return OrderResult(False, result.retcode if result else -1, name)
        logger.info("Modified position %s -> sl=%.5f tp=%.5f", ticket, sl, tp)
        return OrderResult(True, result.retcode, "done", ticket=ticket)

    async def close_position(self, position) -> OrderResult:
        """Close an open position with a market order in the opposite direction."""
        info = await self.engine.call(mt5.symbol_info, position.symbol)
        if info is None:
            return OrderResult(False, -1, "no symbol info",
                               message=f"symbol_info({position.symbol}) failed")
        is_buy = position.direction == 1
        price = await self._live_price(position.symbol, -1 if is_buy else 1)
        if price is None:
            return OrderResult(False, -1, "no tick", message="no live price")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,   # targets THIS exact position
            "price": float(price),
            "deviation": self.config.DEVIATION_POINTS,
            "magic": self.config.MAGIC_NUMBER,
            "comment": "bot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(SymbolInfoLike(info)),
        }
        result = await self._submit(request, SymbolInfoLike(info), -1 if is_buy else 1)
        if result.success:
            logger.info("CLOSE %s ticket=%s vol=%.2f @ %.5f",
                        position.symbol, position.ticket, position.volume, price)
        return result


class SymbolInfoLike:
    """Minimal shim so _submit/clamp_stops work for close_position too."""
    def __init__(self, info):
        self.name = info.name
        self.point = info.point
        self.tick_size = info.trade_tick_size
        self.tick_value = info.trade_tick_value
        self.volume_min = info.volume_min
        self.volume_max = info.volume_max
        self.volume_step = info.volume_step
        self.stops_level = info.trade_stops_level
        self.filling_mode = info.filling_mode
