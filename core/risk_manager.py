"""
RiskManager – position sizing, SL/TP, margin verification, trailing stops,
breakeven management, partial-close logic, and daily drawdown tracking.

New in v2 (AI edition)
-----------------------
  • Dynamic risk scaling   – scale RISK_PER_TRADE up/down with ADX strength
  • Breakeven management   – move SL to entry once profit reaches 1 ATR
  • Partial-close trigger  – signal a 50 % close at 1.5 ATR profit
  • DailyPnLTracker        – per-session equity gate, resets each calendar day

Fixed-fractional position sizing (unchanged)
--------------------------------------------
    risk_amount  = equity × RISK_PER_TRADE × risk_scale(ADX)
    sl_distance  = |entry − SL| / tick_size          (distance in ticks)
    risk_per_lot = sl_distance × tick_value
    volume       = ⌊risk_amount / risk_per_lot⌋ rounded to volume_step,
                   clamped to [volume_min, volume_max]

Margin verification (unchanged)
--------------------------------
    Before every order we compute mt5.order_calc_margin and reject the trade
    if free margin is insufficient or total margin usage > MAX_MARGIN_USAGE.

ATR trailing stop (unchanged)
-------------------------------
    long:  new_sl = max(current_sl,  close − TRAIL_ATR_MULT × ATR)
    short: new_sl = min(current_sl,  close + TRAIL_ATR_MULT × ATR)
    Only activates after TRAIL_ACTIVATE_ATR × ATR of profit.
"""
import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

_EP = 1e-10   # epsilon


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SizingResult:
    volume: float       = 0.0
    sl: float           = 0.0
    tp: float           = 0.0
    risk_amount: float  = 0.0
    margin_needed: float = 0.0
    rejected: bool      = False
    reason: str         = ""


@dataclass
class BreakevenResult:
    should_move: bool   = False
    new_sl: float       = 0.0


@dataclass
class PartialCloseResult:
    should_close: bool  = False
    volume: float       = 0.0   # volume to close


# ---------------------------------------------------------------------------
# Daily P&L tracker
# ---------------------------------------------------------------------------

class DailyPnLTracker:
    """
    Tracks the equity at the start of each calendar day and checks whether
    the configured maximum daily drawdown has been breached.

    Call `update(equity)` once per poll cycle; `is_limit_hit()` will return
    True only after the daily drop exceeds MAX_DAILY_DRAWDOWN_PCT × start balance.
    """

    def __init__(self):
        self._start_equity: float = 0.0
        self._current_date: Optional[date] = None

    def update(self, equity: float) -> None:
        today = date.today()
        if self._current_date != today:
            self._current_date  = today
            self._start_equity  = equity
            logger.info("DailyPnLTracker: new day – start equity=%.2f", equity)

    def is_limit_hit(self, equity: float, max_dd_pct: float) -> bool:
        if self._start_equity <= 0:
            return False
        dd = (self._start_equity - equity) / self._start_equity
        if dd >= max_dd_pct:
            logger.warning(
                "Daily drawdown limit hit: %.2f%% (start=%.2f, now=%.2f)",
                dd * 100, self._start_equity, equity)
            return True
        return False

    @property
    def start_equity(self) -> float:
        return self._start_equity


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.daily_tracker = DailyPnLTracker()

    # ------------------------------------------------------------------
    # Dynamic risk scaling
    # ------------------------------------------------------------------
    def _risk_scale(self, adx: float) -> float:
        """
        Map ADX → a risk multiplier in [MIN_RISK_SCALE, MAX_RISK_SCALE].
        • ADX ≤ ADX_TRENDING_MIN → MIN_RISK_SCALE  (weak/ranging market)
        • ADX ≥ ADX_STRONG_TREND + 20 → MAX_RISK_SCALE (strong trend)
        • Linear interpolation in between.
        """
        c      = self.config
        lo     = getattr(c, "MIN_RISK_SCALE", 0.5)
        hi     = getattr(c, "MAX_RISK_SCALE", 1.5)
        adx_lo = getattr(c, "ADX_TRENDING_MIN", 20)
        adx_hi = getattr(c, "ADX_STRONG_TREND", 30) + 20   # = 50
        if adx <= adx_lo:
            return lo
        if adx >= adx_hi:
            return hi
        t = (adx - adx_lo) / (adx_hi - adx_lo)
        return lo + t * (hi - lo)

    # ------------------------------------------------------------------
    # Stop clamping
    # ------------------------------------------------------------------
    def clamp_stops(self, entry: float, sl: float, tp: float, symbol_info):
        """
        Widen SL/TP if they sit inside the broker's minimum stop distance
        (SYMBOL_TRADE_STOPS_LEVEL × point).  Prevents TRADE_RETCODE_INVALID_STOPS.
        """
        min_dist = symbol_info.stops_level * symbol_info.point
        if min_dist <= 0:
            return sl, tp
        # Long position (entry > sl, tp > entry)
        if abs(entry - sl) < min_dist:
            sl = entry - min_dist if sl < entry else entry + min_dist
        if abs(tp - entry) < min_dist:
            tp = entry + min_dist if tp > entry else entry - min_dist
        return sl, tp

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    async def size_position(
        self, symbol_info, direction: int, entry: float,
        atr: float, equity: float, adx: float = 0.0,
    ) -> SizingResult:
        """
        Compute volume / SL / TP, verify margin, and return a SizingResult.

        Parameters
        ----------
        adx : current ADX value used for dynamic risk scaling (0 = disabled).
        """
        c        = self.config
        sl_dist  = c.SL_ATR_MULT * atr
        if direction == 1:
            sl = entry - sl_dist
            tp = entry + sl_dist * c.TP_RR_RATIO
        else:
            sl = entry + sl_dist
            tp = entry - sl_dist * c.TP_RR_RATIO

        sl, tp = self.clamp_stops(entry, sl, tp, symbol_info)

        # --- fixed-fractional sizing with optional dynamic scaling ------
        scale        = (self._risk_scale(adx)
                        if getattr(c, "DYNAMIC_RISK", False) and adx > 0
                        else 1.0)
        risk_amount  = equity * c.RISK_PER_TRADE * scale
        sl_ticks     = abs(entry - sl) / (symbol_info.tick_size + _EP)
        if sl_ticks <= 0:
            return SizingResult(rejected=True, reason="SL distance ≤ 0")
        risk_per_lot = sl_ticks * symbol_info.tick_value
        if risk_per_lot <= 0:
            return SizingResult(rejected=True, reason="risk_per_lot ≤ 0")

        raw    = risk_amount / risk_per_lot
        step   = symbol_info.volume_step or 0.01
        volume = math.floor(raw / step) * step
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

        if volume < symbol_info.volume_min:
            return SizingResult(rejected=True, reason=(
                f"risk-adjusted volume {volume:.3f} < min lot {symbol_info.volume_min}"))

        # --- margin verification -----------------------------------------
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        margin     = await self.engine.call(
            mt5.order_calc_margin, order_type, symbol_info.name, volume, entry)
        if margin is None or margin <= 0:
            return SizingResult(rejected=True, reason="order_calc_margin failed")

        account = await self.engine.call(mt5.account_info)
        if account is None:
            return SizingResult(rejected=True, reason="account_info failed")
        if margin > account.margin_free:
            return SizingResult(volume, sl, tp, risk_amount, margin, True,
                                f"margin {margin:.2f} > free {account.margin_free:.2f}")
        if account.equity > 0:
            total_margin_usage = (account.margin + margin) / account.equity
            if total_margin_usage > c.MAX_MARGIN_USAGE:
                return SizingResult(volume, sl, tp, risk_amount, margin, True,
                                    f"margin usage {total_margin_usage:.0%} > limit {c.MAX_MARGIN_USAGE:.0%}")

        logger.debug(
            "Sizing %s dir=%+d vol=%.2f sl=%.5f tp=%.5f "
            "risk=%.2f margin=%.2f scale=%.2f adx=%.1f",
            symbol_info.name, direction, volume, sl, tp,
            risk_amount, margin, scale, adx)
        return SizingResult(volume, sl, tp, risk_amount, margin)

    # ------------------------------------------------------------------
    # ATR trailing stop (unchanged logic, moved here for clarity)
    # ------------------------------------------------------------------
    def trailing_stop(
        self, direction: int, close: float, atr: float,
        entry_price: float, current_sl: Optional[float],
    ) -> Optional[float]:
        """
        Return the new SL level if the trailing stop should ratchet, else None.
        The ratchet only moves the stop in the profitable direction.
        """
        c = self.config
        if atr <= 0:
            return None
        profit_move = ((close - entry_price) if direction == 1
                       else (entry_price - close))
        if profit_move < c.TRAIL_ACTIVATE_ATR * atr:
            return None
        atr_dist = c.TRAIL_ATR_MULT * atr
        if direction == 1:
            new_sl = close - atr_dist
            return new_sl if (current_sl is None or new_sl > current_sl) else None
        new_sl = close + atr_dist
        return new_sl if (current_sl is None or new_sl < current_sl) else None

    # ------------------------------------------------------------------
    # Breakeven management
    # ------------------------------------------------------------------
    def check_breakeven(
        self, direction: int, entry: float, current_sl: Optional[float],
        close: float, atr: float,
    ) -> BreakevenResult:
        """
        Return BreakevenResult(should_move=True, new_sl=entry) when the
        position has profited by at least BREAKEVEN_TRIGGER_ATR × ATR and the
        SL has not yet been moved to / beyond the entry price.
        """
        c = self.config
        if not getattr(c, "ENABLE_BREAKEVEN", True) or atr <= 0:
            return BreakevenResult()

        # Already at or beyond breakeven?
        if current_sl is not None:
            if direction == 1  and current_sl >= entry:
                return BreakevenResult()
            if direction == -1 and current_sl <= entry:
                return BreakevenResult()

        profit_atr = ((close - entry) if direction == 1
                      else (entry - close)) / atr
        if profit_atr < getattr(c, "BREAKEVEN_TRIGGER_ATR", 1.0):
            return BreakevenResult()

        return BreakevenResult(should_move=True, new_sl=entry)

    # ------------------------------------------------------------------
    # Partial-close trigger
    # ------------------------------------------------------------------
    def check_partial_close(
        self, direction: int, entry: float, volume: float,
        close: float, atr: float, volume_step: float, volume_min: float,
    ) -> PartialCloseResult:
        """
        Return PartialCloseResult(should_close=True, volume=X) when the
        position has profited by PARTIAL_CLOSE_TRIGGER_ATR × ATR.
        The returned volume is rounded to the broker lot step.
        """
        c = self.config
        if not getattr(c, "ENABLE_PARTIAL_CLOSE", True) or atr <= 0:
            return PartialCloseResult()

        profit_atr = ((close - entry) if direction == 1
                      else (entry - close)) / atr
        trig = getattr(c, "PARTIAL_CLOSE_TRIGGER_ATR", 1.5)
        if profit_atr < trig:
            return PartialCloseResult()

        fraction   = getattr(c, "PARTIAL_CLOSE_FRACTION", 0.5)
        step       = volume_step or 0.01
        raw_vol    = volume * fraction
        partial    = math.floor(raw_vol / step) * step
        # Ensure enough volume remains open after the partial close
        remaining  = volume - partial
        if partial < volume_min or remaining < volume_min:
            return PartialCloseResult()

        return PartialCloseResult(should_close=True, volume=partial)
