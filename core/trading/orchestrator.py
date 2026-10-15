"""
core/trading/orchestrator.py — The missing wire between analysis and orders.

Before this module, ``core/`` contained a complete strategy stack and a
complete risk stack that nothing connected: ``StrategyEnsemble`` and
``MarketIntelligenceEngine`` were constructed only by tests, and ``run.py``
imported none of ``core/``. The bot could compute an opinion and size a trade
and never once have to reconcile the two, let alone act.

This is the component that does. The loop is deliberately linear and every
stage can refuse:

    candles -> location -> signal -> gate -> stop -> target -> risk -> execution

Each arrow is a place where the trade dies if the information does not justify
it, and each refusal is recorded with its reason rather than swallowed. The
returned ``TradingDecision`` is a complete audit trail: what the market looked
like, what the strategies thought, what location vetoed, where the stop went
and what risk decided. That object is the difference between a bot and a
black box, and it is what should be logged on every cycle.

Ordering is a safety property, not a style choice. Location is computed
*before* the signal is acted on so a veto is cheap, and the structural stop is
derived *before* risk approval so that the lot size the risk engine computes
reflects the real distance to invalidation rather than a blind ATR multiple.
A tighter, honest stop correctly produces a larger size for the same risk —
which is only safe because the stop is genuinely where the market says the
idea is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from core.market.engine import MarketEngine
from core.market.location import MarketLocation, StopPlan, analyse_location
from core.mt5.adapter import MT5Adapter
from core.mt5.types import AccountInfo, Candle, SymbolInfo, Tick, Timeframe
from core.risk.engine import RiskDecisionResult, RiskEngine, TradeProposal
from core.strategy.base import Signal
from core.strategy.ensemble import StrategyEnsemble

logger = logging.getLogger(__name__)


class Rejection(str):
    """Stable, greppable rejection codes. Never free-text at the call site."""


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradingDecision:
    """Everything that happened on one symbol, and why."""

    symbol: str
    time: datetime
    action: str                       # EXECUTED | REJECTED | NO_SIGNAL
    reason: str
    location: Optional[MarketLocation] = None
    signal: Optional[Signal] = None
    gate_reason: str = ""
    proposal: Optional[TradeProposal] = None
    stop_plan: Optional[StopPlan] = None
    risk: Optional[RiskDecisionResult] = None
    order_result: Optional[object] = None
    lot_size: float = 0.0

    @property
    def executed(self) -> bool:
        return self.action == "EXECUTED"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "time": self.time.isoformat(),
            "action": self.action,
            "reason": self.reason,
            "gate_reason": self.gate_reason,
            "lot_size": self.lot_size,
            "location": self.location.to_dict() if self.location else None,
            "signal": {
                "direction": self.signal.direction,
                "score": self.signal.score,
                "reason": self.signal.reason,
                "invalidation": self.signal.invalidation,
            } if self.signal else None,
            "stop_plan": self.stop_plan.to_dict() if self.stop_plan else None,
            "risk": {
                "verdict": self.risk.verdict.value,
                "reason": self.risk.reason,
                "passed": self.risk.checks_passed,
                "failed": self.risk.checks_failed,
                "lot_size": self.risk.calculated_lot_size,
            } if self.risk else None,
        }

    def __str__(self) -> str:
        where = self.location.describe() if self.location else "no location"
        return f"[{self.symbol}] {self.action}: {self.reason} | {where}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TradingOrchestrator:
    """
    Drives one symbol from candles to, optionally, an order.

    Args:
        adapter: Broker adapter. ``MockMT5Adapter`` makes the whole loop
            runnable and testable without Windows or MT5.
        risk_engine: The authority on whether a trade may proceed.
        execution_engine: Sends approved proposals.
        ensemble: Produces the signal ("what").
        execute: When ``False`` the loop runs all the way through risk
            approval and stops short of sending. This is the default-safe
            posture for a paper run: the analysis is identical, only the
            broker is left out.
        timeframe: Analysis timeframe.
        min_score: Signal score floor, on top of whatever the ensemble decides.
        use_atr_fallback_target: When no liquidity pool sits beyond entry in the
            trade's favour, fall back to a fixed R multiple. ``False`` instead
            refuses the trade, which is the stricter and more honest option.
        rr_fallback: R multiple used by the fallback target.
    """

    def __init__(
        self,
        adapter: MT5Adapter,
        risk_engine: RiskEngine,
        execution_engine,
        ensemble: StrategyEnsemble,
        execute: bool = False,
        timeframe: Timeframe = Timeframe.M15,
        candles: int = 500,
        min_score: float = 55.0,
        use_atr_fallback_target: bool = True,
        rr_fallback: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.ensemble = ensemble
        self.execute = execute
        self.timeframe = timeframe
        self.candles = candles
        self.min_score = min_score
        self.use_atr_fallback_target = use_atr_fallback_target
        self.rr_fallback = rr_fallback

        self.market_engine = MarketEngine(adapter, symbols=[])
        self.history: List[TradingDecision] = []

    # ── the loop ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        account: Optional[AccountInfo] = None,
        symbol_info: Optional[SymbolInfo] = None,
    ) -> TradingDecision:
        """Run one full analysis cycle for ``symbol``. Never raises."""
        now = datetime.now(timezone.utc)

        try:
            df = self.market_engine.get_data(symbol, self.timeframe, count=self.candles)
        except Exception as exc:  # pragma: no cover - broker/network failure
            logger.exception("orchestrator: data fetch failed for %s", symbol)
            return self._record(symbol, now, "REJECTED", f"data_fetch_failed: {exc}")

        if df is None or df.empty:
            return self._record(symbol, now, "REJECTED", "no_market_data")

        # 1. Where is the market?
        location = analyse_location(df)
        if not location.is_valid:
            return self._record(
                symbol, now, "REJECTED",
                "location_undefined: no confirmed range to trade against",
                location=location,
            )

        # 2. What do the strategies think?
        signal = self.ensemble.evaluate_all(df)
        if signal.direction == 0:
            return self._record(symbol, now, "NO_SIGNAL", signal.reason or "neutral", location, signal)

        if signal.score < self.min_score:
            return self._record(
                symbol, now, "NO_SIGNAL",
                f"score {signal.score:.1f} below floor {self.min_score:.1f}",
                location, signal,
            )

        # 3. Is this direction sane here?
        gate = location.entry_gate(signal.direction)
        if not gate.allowed:
            return self._record(
                symbol, now, "REJECTED", f"location_veto: {gate.reason}",
                location, signal,
            )

        # 4. Where does the idea stop being true?
        tick = self.adapter.get_tick(symbol)
        if tick is None:
            return self._record(symbol, now, "REJECTED", "no_tick", location, signal)

        entry = tick.ask if signal.direction > 0 else tick.bid

        # Broker contract metadata. ``RiskEngine.calculate_lot_size`` refuses to
        # size without it, by design, so the orchestrator sources it from the
        # adapter rather than relying on every caller to remember. It is never
        # defaulted: an absent ``point`` or ``tick_size`` stays ``None`` and
        # propagates to the risk engine as absence.
        if symbol_info is None:
            try:
                symbol_info = self.adapter.get_symbol_info(symbol)
            except Exception as exc:  # pragma: no cover - broker failure
                logger.warning(
                    "orchestrator: symbol info unavailable for %s: %s", symbol, exc
                )
                symbol_info = None

        spec_tick = _tick_size(symbol_info)

        stop_plan = location.structural_stop(
            entry, signal.direction, tick_size=spec_tick,
        )
        if stop_plan is None:
            # No defensible stop means no trade. Never fall back to a default.
            return self._record(
                symbol, now, "REJECTED",
                "no_structural_stop: refusing to trade without a valid invalidation",
                location, signal,
            )

        target, target_reason = self._target_for(location, signal.direction, entry, stop_plan)
        if target is None:
            return self._record(
                symbol, now, "REJECTED",
                f"no_target: {target_reason}", location, signal,
            )

        try:
            positions = self.adapter.get_positions(symbol=symbol) or []
        except Exception as exc:  # pragma: no cover - broker failure
            logger.warning("orchestrator: position lookup failed for %s: %s", symbol, exc)
            return self._record(
                symbol, now, "REJECTED", f"position_lookup_failed: {exc}",
                location, signal, stop_plan=stop_plan,
            )

        open_positions = len(positions)
        used_margin = sum((getattr(p, "margin", 0.0) or 0.0) for p in positions)

        # 5. Risk has the final word.
        #
        # No try/except here on purpose. ``approve`` documents that it never
        # raises, so an exception escaping it is a programming error — a wrong
        # signature or a None where a number is required — and disguising it as
        # a risk rejection turns a bug into a silent "no trade". It must be
        # loud during development, not a quiet reason string in production.
        proposal = TradeProposal(
            symbol=symbol,
            direction=signal.direction,
            entry_price=entry,
            stop_loss=stop_plan.price,
            take_profit=target,
            strategy_name=signal.reason[:120] or "ensemble",
            signal_score=signal.score,
            account_info=account,
            symbol_info=symbol_info,
            tick=tick,
        )

        adx = 0.0
        if "adx" in df.columns and len(df):
            candidate = df["adx"].iloc[-1]
            if candidate is not None and candidate == candidate:  # not NaN
                adx = float(candidate)

        risk_result = self.risk_engine.approve(
            proposal,
            open_positions_count=open_positions,
            adx=adx,
            used_margin=used_margin,
        )

        if not risk_result.approved:
            return self._record(
                symbol, now, "REJECTED", f"risk: {risk_result.reason}",
                location, signal, proposal, stop_plan, risk_result,
            )

        if not self.execute:
            return self._record(
                symbol, now, "REJECTED", "approved_but_execution_disabled (paper run)",
                location, signal, proposal, stop_plan, risk_result,
                lot_size=risk_result.calculated_lot_size,
            )

        # 6. Send it.
        order_result = self.execution_engine.execute_proposal(
            proposal, risk_result.calculated_lot_size
        )

        success = getattr(order_result, "success", False)
        return self._record(
            symbol, now,
            "EXECUTED" if success else "REJECTED",
            "order_filled" if success else f"order_failed: {getattr(order_result, 'comment', '')}",
            location, signal, proposal, stop_plan, risk_result,
            order_result=order_result, lot_size=risk_result.calculated_lot_size,
        )

    # ── target selection ────────────────────────────────────────────────────

    def _target_for(
        self,
        location: MarketLocation,
        direction: int,
        entry: float,
        stop: StopPlan,
    ) -> tuple[Optional[float], str]:
        """
        Prefer a real liquidity pool as the objective.

        A resting pool of the right side is where the stops are, so it is both
        the natural magnet and the most honest target available. Only when no
        such pool exists do we fall back to a fixed R multiple — and that
        fallback is configurable off for callers who would rather abstain.
        """
        pool = location.target_above() if direction > 0 else location.target_below()
        if pool is not None:
            target = pool.price
            # Leave a small margin inside the pool: aiming exactly at the
            # liquidity means aiming at the price where it evaporates.
            if location.atr:
                target -= direction * location.atr * 0.10
            if (target - entry) * direction > 0:
                return target, f"liquidity pool {pool.kind.value} @ {pool.price:.5f}"

        if not self.use_atr_fallback_target:
            return None, "no untaken liquidity pool in the trade's favour"

        if not location.atr:
            return None, "no volatility scale for an R-multiple target"

        distance = abs(entry - stop.price)
        if distance <= 0:
            return None, "degenerate stop distance"

        return entry + direction * distance * self.rr_fallback, f"{self.rr_fallback}R fallback"

    # ── helpers ─────────────────────────────────────────────────────────────

    def _record(
        self,
        symbol: str,
        now: datetime,
        action: str,
        reason: str,
        location: Optional[MarketLocation] = None,
        signal: Optional[Signal] = None,
        proposal: Optional[TradeProposal] = None,
        stop_plan: Optional[StopPlan] = None,
        risk: Optional[RiskDecisionResult] = None,
        order_result=None,
        lot_size: float = 0.0,
    ) -> TradingDecision:
        decision = TradingDecision(
            symbol=symbol,
            time=now,
            action=action,
            reason=reason,
            location=location,
            signal=signal,
            gate_reason=(signal.reason if signal else ""),
            proposal=proposal,
            stop_plan=stop_plan,
            risk=risk,
            order_result=order_result,
            lot_size=lot_size,
        )
        self.history.append(decision)
        logger.info("%s", decision)
        return decision

    def evaluate_all(self, symbols: List[str], **kwargs) -> List[TradingDecision]:
        """Run a cycle across symbols. Each symbol fails independently."""
        return [self.evaluate(symbol, **kwargs) for symbol in symbols]

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for decision in self.history:
            counts[decision.action] = counts.get(decision.action, 0) + 1
        return counts


def _tick_size(symbol_info: Optional[SymbolInfo]) -> Optional[float]:
    """
    Broker price increment, or ``None`` when the broker did not supply one.

    Read defensively with ``getattr``: ``SymbolInfo`` names the field
    ``trade_tick_size``, not ``tick_size``, and a typo here would raise
    ``AttributeError`` on every evaluation rather than degrading to "unknown
    increment". Absence is the honest answer and levels stay unsnapped.
    """
    if symbol_info is None:
        return None
    for field_name in ("trade_tick_size", "tick_size", "point"):
        value = getattr(symbol_info, field_name, None)
        if value:
            return float(value)
    return None