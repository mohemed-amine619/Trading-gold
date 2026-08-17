"""
RiskManager - position sizing, SL/TP placement, margin verification and
ATR-based trailing stops.

Fixed-fractional position sizing
--------------------------------
We risk a fixed fraction of CURRENT equity per trade:
    risk_amount  = equity * RISK_PER_TRADE
    sl_distance  = |entry - SL| / tick_size           (distance in ticks)
    risk_per_lot = sl_distance * tick_value           (account-ccy loss for
                                                       1.0 lot if SL is hit,
                                                       because one tick move
                                                       on 1 lot is worth
                                                       exactly `tick_value`)
    volume       = floor(risk_amount / risk_per_lot) down to volume_step,
                   clamped to [volume_min, volume_max]

Margin verification
-------------------
Before every order we compute the margin the trade would consume
(mt5.order_calc_margin) and reject the trade if free margin is insufficient
or total margin usage would exceed MAX_MARGIN_USAGE.

ATR trailing stop
-----------------
Once the price has moved TRAIL_ACTIVATE_ATR * ATR in our favour, the stop
ratchets behind the price at a fixed ATR distance (never backwards):
    long:  new_sl = max(current_sl, close - TRAIL_ATR_MULT * ATR)
    short: new_sl = min(current_sl, close + TRAIL_ATR_MULT * ATR)
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    volume: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    risk_amount: float = 0.0
    margin_needed: float = 0.0
    rejected: bool = False
    reason: str = ""


class RiskManager:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config

    def clamp_stops(self, entry: float, sl: float, tp: float, symbol_info):
        """Widen SL/TP if they sit inside the broker's minimum stop distance.

        Brokers reject orders whose SL/TP is closer than SYMBOL_TRADE_STOPS_LEVEL
        (expressed in points) -> they would otherwise answer TRADE_RETCODE_
        INVALID_STOPS (10016).
        """
        min_dist = symbol_info.stops_level * symbol_info.point
        if min_dist <= 0:
            return sl, tp
        if entry - sl < min_dist:      # long SL too close to price
            sl = entry - min_dist
        if tp - entry < min_dist:      # long TP too close to price
            tp = entry + min_dist
        return sl, tp

    async def size_position(self, symbol_info, direction: int, entry: float,
                            atr: float, equity: float) -> SizingResult:
        """Compute volume / SL / TP for a trade, verifying margin beforehand."""
        c = self.config
        sl_dist_price = c.SL_ATR_MULT * atr
        if direction == 1:  # long
            sl = entry - sl_dist_price
            tp = entry + sl_dist_price * c.TP_RR_RATIO
        else:               # short
            sl = entry + sl_dist_price
            tp = entry - sl_dist_price * c.TP_RR_RATIO

        sl, tp = self.clamp_stops(entry, sl, tp, symbol_info)

        # --- fixed fractional sizing ---------------------------------------
        risk_amount = equity * c.RISK_PER_TRADE
        sl_ticks = abs(entry - sl) / symbol_info.tick_size
        if sl_ticks <= 0:
            return SizingResult(rejected=True, reason="SL distance <= 0")
        risk_per_lot = sl_ticks * symbol_info.tick_value
        if risk_per_lot <= 0:
            return SizingResult(rejected=True, reason="risk_per_lot <= 0")

        raw = risk_amount / risk_per_lot
        step = symbol_info.volume_step or 0.01
        # Round DOWN to the broker's lot step: never oversize on the margin.
        volume = math.floor(raw / step) * step
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

        if volume < symbol_info.volume_min:
            return SizingResult(rejected=True, reason=(
                f"risk-adjusted volume {volume:.3f} < min lot {symbol_info.volume_min}"))

        # --- margin verification BEFORE the order is sent ------------------
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        margin = await self.engine.call(
            mt5.order_calc_margin, order_type, symbol_info.name, volume, entry)
        if margin is None or margin <= 0:
            return SizingResult(rejected=True, reason="order_calc_margin failed")

        account = await self.engine.call(mt5.account_info)
        if account is None:
            return SizingResult(rejected=True, reason="account_info failed")
        if margin > account.margin_free:
            return SizingResult(volume, sl, tp, risk_amount, margin, True,
                                f"margin {margin:.2f} > free {account.margin_free:.2f}")
        if account.equity > 0 and (account.margin + margin) / account.equity > c.MAX_MARGIN_USAGE:
            return SizingResult(volume, sl, tp, risk_amount, margin, True,
                                f"margin usage would exceed {c.MAX_MARGIN_USAGE:.0%}")

        logger.debug("Sizing %s dir=%+d vol=%.2f sl=%.5f tp=%.5f risk=%.2f margin=%.2f",
                     symbol_info.name, direction, volume, sl, tp, risk_amount, margin)
        return SizingResult(volume, sl, tp, risk_amount, margin)

    def trailing_stop(self, direction: int, close: float, atr: float,
                      entry_price: float, current_sl: Optional[float]) -> Optional[float]:
        """Return the NEW stop level if the trailing stop should move, else None.

        `current_sl` is the position's SL (or None). The ratchet only moves the
        stop in the profitable direction, so a trailing stop can never be
        pulled back toward the entry.
        """
        c = self.config
        if atr <= 0:
            return None
        profit_move = (close - entry_price) if direction == 1 else (entry_price - close)
        # Activate trailing only after a meaningful move (filters noise).
        if profit_move < c.TRAIL_ACTIVATE_ATR * atr:
            return None

        atr_dist = c.TRAIL_ATR_MULT * atr
        if direction == 1:
            new_sl = close - atr_dist
            return new_sl if (current_sl is None or new_sl > current_sl) else None
        new_sl = close + atr_dist
        return new_sl if (current_sl is None or new_sl < current_sl) else None
