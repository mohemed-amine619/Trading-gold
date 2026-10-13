"""
core/risk/engine.py — Risk Engine: the system's hard limits.

The RiskEngine is the absolute authority on whether a trade may proceed.
The AI, strategies, and user preferences cannot override it.

Hard limits enforced:
  - Risk per trade (max fraction of equity lost if SL hit)
  - Daily loss limit (stop trading for the day)
  - Weekly loss limit
  - Maximum drawdown
  - Maximum simultaneous open positions
  - Maximum exposure per symbol
  - Maximum spread
  - Minimum risk:reward ratio
  - Stop-loss requirement (every trade must have one)
  - Session restrictions
  - Consecutive loss streak cooldown
  - Margin limit
  - Emergency kill switch
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from core.mt5.types import AccountInfo, SymbolInfo, Tick

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

class RiskVerdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class RiskDecisionResult:
    verdict: RiskVerdict
    reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    calculated_lot_size: float = 0.0
    calculated_sl: float = 0.0
    calculated_tp: float = 0.0
    current_daily_pnl: float = 0.0
    current_drawdown_pct: float = 0.0
    current_open_positions: int = 0

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED


@dataclass
class TradeProposal:
    """Input to the risk engine for a potential trade."""
    symbol: str
    direction: int          # +1 BUY | -1 SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    ai_confidence: float = 0.0
    signal_score: float = 0.0
    account_info: Optional[AccountInfo] = None
    symbol_info: Optional[SymbolInfo] = None
    tick: Optional[Tick] = None
    # Margin this order will require, in account currency. Callers that can
    # compute it (via MT5Adapter.calc_margin) should; 0.0 is treated as
    # "unknown" by the margin check, which then relies on committed margin
    # alone rather than assuming a free margin allowance.
    margin_estimate: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine
# ─────────────────────────────────────────────────────────────────────────────

# Parameters that may be changed at runtime, with their permitted ranges.
# Anything not listed here is internal state and must not be writable from an
# API request — see RiskEngine.update_config.
_CONFIGURABLE: dict[str, tuple[float, float]] = {
    "risk_per_trade": (0.001, 0.05),
    "max_daily_loss_pct": (0.005, 0.20),
    "max_weekly_loss_pct": (0.005, 0.30),
    "max_drawdown_pct": (0.01, 0.50),
    "max_open_positions": (1, 50),
    "min_rr_ratio": (0.5, 10.0),
    "max_margin_usage": (0.10, 0.90),
    "max_spread_points": (0.0, 10_000.0),
    "max_consecutive_losses": (1, 100),
    "loss_streak_cooldown_minutes": (0, 1440),
    "max_trades_per_day": (1, 500),
    "dynamic_risk": (0, 1),  # handled as bool below
    "min_risk_scale": (0.0, 1.0),
    "max_risk_scale": (0.0, 3.0),
    "adx_trending_min": (0.0, 100.0),
    "adx_strong_trend": (0.0, 100.0),
}


class RiskEngine:
    """
    Stateful risk engine.

    Tracks session-level state: daily P&L, equity peak, consecutive losses,
    daily trade count.

    All checks are independent of the AI.  If the AI becomes unavailable,
    the risk engine still protects existing positions through deterministic
    rules.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_daily_loss_pct: float = 0.02,
        max_weekly_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
        max_open_positions: int = 5,
        min_rr_ratio: float = 1.5,
        max_margin_usage: float = 0.50,
        max_spread_points: Optional[float] = None,
        max_consecutive_losses: int = 5,
        loss_streak_cooldown_minutes: int = 60,
        max_trades_per_day: int = 8,
        sl_atr_mult: float = 2.0,
        tp_rr_ratio: float = 2.5,
        dynamic_risk: bool = True,
        min_risk_scale: float = 0.5,
        max_risk_scale: float = 1.5,
        adx_trending_min: float = 20.0,
        adx_strong_trend: float = 30.0,
    ) -> None:
        # Configuration
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.min_rr_ratio = min_rr_ratio
        self.max_margin_usage = max_margin_usage
        self.max_spread_points = max_spread_points
        self.max_consecutive_losses = max_consecutive_losses
        self.loss_streak_cooldown_minutes = loss_streak_cooldown_minutes
        self.max_trades_per_day = max_trades_per_day
        self.sl_atr_mult = sl_atr_mult
        self.tp_rr_ratio = tp_rr_ratio
        self.dynamic_risk = dynamic_risk
        self.min_risk_scale = min_risk_scale
        self.max_risk_scale = max_risk_scale
        self.adx_trending_min = adx_trending_min
        self.adx_strong_trend = adx_strong_trend

        # State
        self._lock = threading.Lock()
        self._emergency_stop = False
        self._equity_peak = 0.0
        self._session_start_equity = 0.0
        self._daily_start_equity = 0.0
        self._daily_date: Optional[date] = None
        self._week_start_equity = 0.0
        self._week_date: Optional[date] = None
        self._consecutive_losses = 0
        self._loss_streak_cooldown_until = 0.0   # epoch seconds
        self._trades_today = 0
        self._trade_date: Optional[date] = None
        self._session_closed_pnl = 0.0  # sum of all closed trade P&L this session

        # Per-symbol open position count (updated by monitor)
        self._open_positions_count = 0

    # ── Configuration reload ─────────────────────────────────────────────────

    def update_config(self, **kwargs) -> None:
        """Runtime reconfiguration from dashboard/API settings.

        Only the keys in ``_CONFIGURABLE`` may be written, and each value must
        validate against its own bound. This method previously used
        ``setattr(self, k, v) for any k that happens to exist``, which meant an
        unauthenticated caller could overwrite ``_emergency_stop`` (disarming
        the kill switch) or ``_daily_start_equity`` (zeroing the daily loss
        limit) through the config endpoint.

        Raises ValueError on an unknown key or an out-of-range value; the
        config is left untouched.
        """
        with self._lock:
            for k, v in kwargs.items():
                if k not in _CONFIGURABLE:
                    raise ValueError(
                        f"{k!r} is not a configurable risk parameter. "
                        f"Configurable keys: {sorted(_CONFIGURABLE)}"
                    )
                bounds = _CONFIGURABLE[k]
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ValueError(f"{k} must be a number, got {type(v).__name__}")
                lo, hi = bounds
                if not (lo <= v <= hi):
                    raise ValueError(f"{k}={v} outside permitted range [{lo}, {hi}]")
            for k, v in kwargs.items():
                setattr(self, k, v)
        logger.info("Risk config updated: %s", {k: kwargs[k] for k in sorted(kwargs)})

    # ── State updates (called by position monitor) ───────────────────────────

    def update_equity(self, balance: float, floating_pnl: float = 0.0) -> None:
        """Called every cycle with latest account balance and real-time floating PNL from the new engine."""
        with self._lock:
            equity = balance + floating_pnl
            today = date.today()
            week_start = today.isocalendar()[1]

            # Daily reset
            if self._daily_date != today:
                self._daily_date = today
                self._daily_start_equity = equity
                self._trades_today = 0
                self._trade_date = today

            # Weekly reset
            if self._week_date is None or self._week_date.isocalendar()[1] != week_start:
                self._week_date = today
                self._week_start_equity = equity

            # Peak tracking
            if equity > self._equity_peak:
                self._equity_peak = equity

            if self._session_start_equity == 0.0:
                self._session_start_equity = equity

    def update_open_positions_count(self, count: int) -> None:
        with self._lock:
            self._open_positions_count = count

    def record_trade_result(self, pnl: float) -> None:
        """Called when a trade closes.  Updates streak and session P&L."""
        with self._lock:
            self._session_closed_pnl += pnl
            if pnl < -1e-9:
                self._consecutive_losses += 1
                if self._consecutive_losses >= self.max_consecutive_losses:
                    cooldown_secs = self.loss_streak_cooldown_minutes * 60
                    self._loss_streak_cooldown_until = time.time() + cooldown_secs
                    logger.warning(
                        "Loss streak of %d reached — cooldown %d minutes",
                        self._consecutive_losses,
                        self.loss_streak_cooldown_minutes,
                    )
            else:
                # Reset streak on a win or breakeven
                self._consecutive_losses = 0

    # ── Emergency kill switch ────────────────────────────────────────────────

    def trigger_emergency_stop(self) -> None:
        with self._lock:
            self._emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED")

    def reset_emergency_stop(self) -> None:
        with self._lock:
            self._emergency_stop = False
        logger.info("Emergency stop reset.")

    @property
    def emergency_stopped(self) -> bool:
        with self._lock:
            return self._emergency_stop

    # ── Main approval gate ───────────────────────────────────────────────────

    def approve(
        self,
        proposal: TradeProposal,
        open_positions_count: int,
        adx: float = 0.0,
        used_margin: float = 0.0,
    ) -> RiskDecisionResult:
        """
        Evaluate a trade proposal against all risk limits.

        Returns a RiskDecisionResult with verdict APPROVED or REJECTED.
        Rejection always includes the specific reason.

        This method NEVER raises — it always returns a result.
        """
        passed: list[str] = []
        failed: list[str] = []

        with self._lock:
            equity = proposal.account_info.equity if proposal.account_info else self._session_start_equity
            # If we don't have account info, we could use a cached self._last_equity if we added it,
            # but using session start equity or an already updated peak is fine.
            daily_start = self._daily_start_equity or equity
            week_start = self._week_start_equity or equity
            peak = self._equity_peak or equity

        def reject(reason: str) -> RiskDecisionResult:
            failed.append(reason)
            logger.info("Risk REJECTED [%s]: %s", proposal.symbol, reason)
            return RiskDecisionResult(
                verdict=RiskVerdict.REJECTED,
                reason=reason,
                checks_passed=passed,
                checks_failed=failed,
                current_daily_pnl=(equity - daily_start),
                current_drawdown_pct=((peak - equity) / peak) if peak > 0 else 0.0,
                current_open_positions=open_positions_count,
            )

        # ── 1. Emergency stop ──────────────────────────────────────────────
        with self._lock:
            if self._emergency_stop:
                return reject("emergency stop active")
        passed.append("emergency_stop_ok")

        # ── 2. Stop-loss required ──────────────────────────────────────────
        if not proposal.stop_loss or proposal.stop_loss == 0.0:
            return reject("stop_loss_required: every trade must have a SL")
        passed.append("sl_required_ok")

        # ── 3. SL direction valid ─────────────────────────────────────────
        if proposal.direction == 1 and proposal.stop_loss >= proposal.entry_price:
            return reject("sl_invalid: BUY SL must be below entry")
        if proposal.direction == -1 and proposal.stop_loss <= proposal.entry_price:
            return reject("sl_invalid: SELL SL must be above entry")
        passed.append("sl_direction_ok")

        # ── 4. Risk:Reward ratio ──────────────────────────────────────────
        sl_dist = abs(proposal.entry_price - proposal.stop_loss)
        tp_dist = abs(proposal.take_profit - proposal.entry_price) if proposal.take_profit else 0.0
        rr = (tp_dist / sl_dist) if sl_dist > 1e-10 else 0.0
        if rr < self.min_rr_ratio:
            return reject(f"rr_ratio: {rr:.2f} < minimum {self.min_rr_ratio:.2f}")
        passed.append(f"rr_ratio_ok ({rr:.2f})")

        # ── 5. Max open positions ─────────────────────────────────────────
        if open_positions_count >= self.max_open_positions:
            return reject(
                f"max_positions: {open_positions_count}/{self.max_open_positions} already open"
            )
        passed.append("max_positions_ok")

        # ── 6. Max trades per day ─────────────────────────────────────────
        with self._lock:
            today = date.today()
            if self._trade_date != today:
                self._trades_today = 0
                self._trade_date = today
            if self._trades_today >= self.max_trades_per_day:
                return reject(
                    f"max_trades_per_day: {self._trades_today}/{self.max_trades_per_day}"
                )
        passed.append("max_trades_per_day_ok")

        # ── 7. Daily loss limit ───────────────────────────────────────────
        if equity > 0 and daily_start > 0:
            daily_dd = (daily_start - equity) / daily_start
            if daily_dd >= self.max_daily_loss_pct:
                return reject(
                    f"daily_loss_limit: {daily_dd:.2%} >= {self.max_daily_loss_pct:.2%}"
                )
            passed.append(f"daily_loss_ok ({daily_dd:.2%})")

        # ── 8. Weekly loss limit ──────────────────────────────────────────
        if equity > 0 and week_start > 0:
            weekly_dd = (week_start - equity) / week_start
            if weekly_dd >= self.max_weekly_loss_pct:
                return reject(
                    f"weekly_loss_limit: {weekly_dd:.2%} >= {self.max_weekly_loss_pct:.2%}"
                )
            passed.append(f"weekly_loss_ok ({weekly_dd:.2%})")

        # ── 9. Maximum drawdown ───────────────────────────────────────────
        if peak > 0 and equity > 0:
            drawdown = (peak - equity) / peak
            if drawdown >= self.max_drawdown_pct:
                return reject(
                    f"max_drawdown: {drawdown:.2%} >= {self.max_drawdown_pct:.2%}"
                )
            passed.append(f"max_drawdown_ok ({drawdown:.2%})")

        # ── 10. Loss streak cooldown ──────────────────────────────────────
        with self._lock:
            cooldown_remaining = self._loss_streak_cooldown_until - time.time()
        if cooldown_remaining > 0:
            return reject(
                f"loss_streak_cooldown: {cooldown_remaining / 60:.1f} min remaining"
            )
        passed.append("loss_streak_ok")

        # ── 11. Spread check ──────────────────────────────────────────────
        if self.max_spread_points and proposal.tick and proposal.symbol_info:
            point = proposal.symbol_info.point or 1e-5
            spread_pts = proposal.tick.spread / point
            if spread_pts > self.max_spread_points:
                return reject(
                    f"spread_too_wide: {spread_pts:.1f} pts > {self.max_spread_points:.1f} pts"
                )
            passed.append(f"spread_ok ({spread_pts:.1f} pts)")

        # ── 12. Calculate position size ───────────────────────────────────
        # Sizing must never be optional. Previously the whole block was gated on
        # `if proposal.symbol_info and equity > 0:`, so a proposal carrying no
        # broker metadata skipped the block, left lot_size at 0.0 and still
        # returned APPROVED — a fail-open on the most safety-critical value.
        if proposal.symbol_info is None:
            return reject("position_size: missing broker symbol metadata (cannot size)")
        if equity <= 0:
            return reject(f"position_size: unusable equity ({equity:.2f})")

        lot_size = self.calculate_lot_size(
            equity=equity,
            sl_distance=sl_dist,
            symbol_info=proposal.symbol_info,
            adx=adx,
        )
        if lot_size <= 0:
            return reject("position_size: calculated lot size is zero or negative")
        passed.append(f"lot_size_ok ({lot_size:.2f})")

        # ── 13. Margin check ──────────────────────────────────────────────
        if proposal.account_info and equity > 0:
            margin_limit = equity * self.max_margin_usage
            # Count the margin the NEW order will consume, not just what is
            # already committed. Comparing committed-only let the last few
            # orders of a session each pass individually while collectively
            # breaching the cap.
            projected = used_margin + proposal.margin_estimate
            if projected >= margin_limit:
                return reject(
                    f"margin_limit: projected {projected:.2f} >= limit {margin_limit:.2f}"
                )
            passed.append(f"margin_ok (projected={projected:.2f})")

        # ── All checks passed ─────────────────────────────────────────────
        result = RiskDecisionResult(
            verdict=RiskVerdict.APPROVED,
            reason="all_checks_passed",
            checks_passed=passed,
            checks_failed=[],
            calculated_lot_size=lot_size,
            calculated_sl=proposal.stop_loss,
            calculated_tp=proposal.take_profit,
            current_daily_pnl=(equity - daily_start),
            current_drawdown_pct=((peak - equity) / peak) if peak > 0 else 0.0,
            current_open_positions=open_positions_count,
        )
        logger.info(
            "Risk APPROVED [%s]: %d lots | %d checks passed",
            proposal.symbol, lot_size, len(passed),
        )
        return result

    # ── Position size calculation ────────────────────────────────────────────

    def calculate_lot_size(
        self,
        equity: float,
        sl_distance: float,
        symbol_info: SymbolInfo,
        adx: float = 0.0,
    ) -> float:
        """
        Fixed-fractional position sizing:
            lots = (equity × risk% × risk_scale) / (sl_ticks × tick_value)

        Clamped to broker [volume_min, volume_max] with step rounding.

        Returns 0.0 when the trade genuinely cannot be sized. Callers MUST
        reject a 0.0 result — it means "refuse", never "trade the minimum".

        This method refuses rather than substitutes a plausible constant when
        broker metadata is missing. It previously did the opposite
        (``trade_tick_value or 1.0``), which silently invented a tick value
        for instruments the broker never described and produced a confident,
        wrong lot size.
        """
        # Refuse rather than default. Missing broker metadata is not a licence
        # to guess — see core/instruments/model.py CRITICAL_FIELDS.
        if (
            sl_distance <= 0
            or equity <= 0
            or not symbol_info.trade_tick_size
            or not symbol_info.trade_tick_value
            or not symbol_info.volume_step
            or symbol_info.volume_min is None
        ):
            logger.error(
                "Refusing to size %s: incomplete broker metadata "
                "(sl_distance=%.6f equity=%.2f tick_size=%r tick_value=%r "
                "step=%r vol_min=%r)",
                symbol_info.name,
                sl_distance,
                equity,
                symbol_info.trade_tick_size,
                symbol_info.trade_tick_value,
                symbol_info.volume_step,
                symbol_info.volume_min,
            )
            return 0.0

        scale = self._risk_scale(adx) if self.dynamic_risk else 1.0
        risk_amount = equity * self.risk_per_trade * scale

        tick_size  = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        sl_ticks   = sl_distance / tick_size
        risk_per_lot = sl_ticks * tick_value

        if risk_per_lot <= 0:
            return 0.0

        raw_lots  = risk_amount / risk_per_lot
        step      = symbol_info.volume_step
        vol_min   = symbol_info.volume_min
        vol_max   = symbol_info.volume_max

        lots = math.floor(raw_lots / step) * step

        # Never round UP to the broker minimum just to produce a tradeable
        # size — that silently risks more than the configured budget. Refuse.
        if lots < vol_min:
            logger.warning(
                "Refusing to size %s: risk budget supports %.4f lots but broker "
                "minimum is %s. Taking the minimum would exceed the risk budget.",
                symbol_info.name, lots, vol_min,
            )
            return 0.0

        if vol_max is not None:
            lots = min(lots, vol_max)
        return lots

    def _risk_scale(self, adx: float) -> float:
        """Scale risk between min and max based on ADX strength."""
        lo_adx = self.adx_trending_min
        hi_adx = self.adx_strong_trend + 20
        if adx <= lo_adx:
            return self.min_risk_scale
        if adx >= hi_adx:
            return self.max_risk_scale
        return self.min_risk_scale + (adx - lo_adx) / (hi_adx - lo_adx) * (
            self.max_risk_scale - self.min_risk_scale
        )

    # ── Trade recorded ───────────────────────────────────────────────────────

    def on_trade_opened(self) -> None:
        """Call this when a trade is successfully opened."""
        with self._lock:
            self._trades_today += 1

    # ── Status snapshot ──────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            equity_peak = self._equity_peak
            daily_start = self._daily_start_equity
            week_start  = self._week_start_equity
        return {
            "emergency_stop": self._emergency_stop,
            "consecutive_losses": self._consecutive_losses,
            "loss_streak_cooldown_remaining_s": max(
                0.0, self._loss_streak_cooldown_until - time.time()
            ),
            "daily_start_equity": daily_start,
            "equity_peak": equity_peak,
            "week_start_equity": week_start,
            "open_positions_count": self._open_positions_count,
            "trades_today": self._trades_today,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global instance access
# ─────────────────────────────────────────────────────────────────────────────

_risk_engine_instance: Optional[RiskEngine] = None


def risk_engine_from_settings(settings=None) -> RiskEngine:
    """Build a RiskEngine from validated Settings.

    This is the only sanctioned way to construct a production RiskEngine.
    ``get_risk_engine()`` used to call ``RiskEngine()`` with nothing but
    hardcoded defaults, so every risk value in ``.env`` (risk_per_trade,
    max_daily_loss_pct, max_open_positions, min_rr_ratio, max_margin_usage,
    max_spread_points, ...) was silently ignored in production.
    """
    if settings is None:
        from core.settings import get_settings
        settings = get_settings()

    return RiskEngine(
        risk_per_trade=settings.risk_per_trade,
        max_daily_loss_pct=settings.max_daily_loss_pct,
        max_weekly_loss_pct=settings.max_weekly_loss_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
        max_open_positions=settings.max_open_positions,
        min_rr_ratio=settings.min_rr_ratio,
        max_margin_usage=settings.max_margin_usage,
        max_spread_points=settings.max_spread_points,
        max_consecutive_losses=settings.max_consecutive_losses,
        loss_streak_cooldown_minutes=settings.loss_streak_cooldown_minutes,
        max_trades_per_day=settings.max_trades_per_day,
        sl_atr_mult=settings.sl_atr_mult,
        tp_rr_ratio=settings.tp_rr_ratio,
        dynamic_risk=settings.dynamic_risk,
        min_risk_scale=settings.min_risk_scale,
        max_risk_scale=settings.max_risk_scale,
        adx_trending_min=settings.adx_trending_min,
        adx_strong_trend=settings.adx_strong_trend,
    )


def get_risk_engine() -> RiskEngine:
    """Return the global RiskEngine instance, creating one if not yet initialized."""
    global _risk_engine_instance
    if _risk_engine_instance is None:
        _risk_engine_instance = risk_engine_from_settings()
    return _risk_engine_instance


def set_risk_engine(engine: Optional[RiskEngine]) -> None:
    """Explicitly set or reset the global RiskEngine instance."""
    global _risk_engine_instance
    _risk_engine_instance = engine

