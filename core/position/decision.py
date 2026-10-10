from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict
from core.market.intelligence import MarketRegime

class PositionAction(str, Enum):
    HOLD = "HOLD"
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
    TRAIL_STOP = "TRAIL_STOP"
    LOCK_PROFIT = "LOCK_PROFIT"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSE_POSITION = "CLOSE_POSITION"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    NO_ACTION = "NO_ACTION"
    ALERT_USER = "ALERT_USER"

@dataclass
class PositionMetrics:
    ticket: int
    floating_pnl: float
    initial_risk: float
    mfe: float  # Maximum Favorable Excursion in price
    mae: float  # Maximum Adverse Excursion in price
    max_floating_profit: float
    current_rr: float
    drawdown_from_peak: float
    protected_profit: float

@dataclass
class DecisionResult:
    action: PositionAction
    rule_triggered: str
    confidence: float
    requires_authorization: bool
    reasons: list[str]

class PNLDecisionEngine:
    def __init__(self, be_activation_rr: float = 1.0, partial_tp_rr: float = 2.0, trail_activation_rr: float = 1.5):
        self.be_activation_rr = be_activation_rr
        self.partial_tp_rr = partial_tp_rr
        self.trail_activation_rr = trail_activation_rr

    def calculate_metrics(self, ticket: int, open_price: float, current_price: float, sl: float, initial_sl: float, direction: int, volume: float, tick_value: float, tick_size: float, historical_high: float, historical_low: float) -> PositionMetrics:
        price_diff = (current_price - open_price) * direction
        floating_pnl = (price_diff / tick_size) * tick_value * volume
        
        initial_risk_price = abs(open_price - initial_sl) if initial_sl > 0 else 0
        initial_risk = (initial_risk_price / tick_size) * tick_value * volume if initial_risk_price > 0 else 1.0

        if direction == 1:
            mfe_price = max(0, historical_high - open_price)
            mae_price = max(0, open_price - historical_low)
        else:
            mfe_price = max(0, open_price - historical_low)
            mae_price = max(0, historical_high - open_price)

        max_floating_profit = (mfe_price / tick_size) * tick_value * volume
        current_rr = floating_pnl / initial_risk if initial_risk > 0 else 0
        drawdown_from_peak = max_floating_profit - floating_pnl
        
        protected_price = (sl - open_price) * direction if sl > 0 else 0
        protected_profit = max(0, (protected_price / tick_size) * tick_value * volume)

        return PositionMetrics(
            ticket=ticket,
            floating_pnl=floating_pnl,
            initial_risk=initial_risk,
            mfe=mfe_price,
            mae=mae_price,
            max_floating_profit=max_floating_profit,
            current_rr=current_rr,
            drawdown_from_peak=drawdown_from_peak,
            protected_profit=protected_profit
        )

    def evaluate(self, metrics: PositionMetrics, regime: MarketRegime) -> DecisionResult:
        reasons = []
        action = PositionAction.HOLD
        rule = "default_hold"
        req_auth = False
        
        if regime == MarketRegime.POTENTIAL_REVERSAL:
            if metrics.current_rr > 0:
                action = PositionAction.CLOSE_POSITION
                rule = "reversal_detected_in_profit"
                reasons.append("Potential reversal detected while in profit. Secure capital.")
            else:
                action = PositionAction.REDUCE_EXPOSURE
                rule = "reversal_detected_in_loss"
                reasons.append("Potential reversal detected while in loss. Reduce exposure.")
                req_auth = True
        elif metrics.current_rr >= self.partial_tp_rr:
            action = PositionAction.PARTIAL_CLOSE
            rule = "partial_tp_target_reached"
            reasons.append(f"Reached {self.partial_tp_rr}R target.")
        elif metrics.current_rr >= self.trail_activation_rr:
            action = PositionAction.TRAIL_STOP
            rule = "trail_activation"
            reasons.append(f"RR > {self.trail_activation_rr}. Activating trailing stop.")
        elif metrics.current_rr >= self.be_activation_rr and metrics.protected_profit <= 0:
            action = PositionAction.MOVE_TO_BREAKEVEN
            rule = "breakeven_activation"
            reasons.append(f"RR > {self.be_activation_rr}. Securing breakeven.")
            
        # Giveback protection
        if metrics.max_floating_profit > metrics.initial_risk * 2:
            giveback_pct = metrics.drawdown_from_peak / metrics.max_floating_profit
            if giveback_pct > 0.5:
                action = PositionAction.LOCK_PROFIT
                rule = "profit_giveback_protection"
                reasons.append(f"Given back {giveback_pct:.0%} of peak profit. Locking remaining.")
        
        return DecisionResult(
            action=action,
            rule_triggered=rule,
            confidence=0.8,
            requires_authorization=req_auth,
            reasons=reasons
        )
