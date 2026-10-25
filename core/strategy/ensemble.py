"""
core/strategy/ensemble.py — Strategy Ensemble and Scoring.
"""

import logging
from typing import List
import pandas as pd

from core.strategy.base import BaseStrategy, MarketRegime, Signal, detect_regime
from core.strategy.strategies import TrendFollowingStrategy, MomentumStrategy, MeanReversionStrategy, FibonacciStrategy

logger = logging.getLogger(__name__)


class StrategyEnsemble:
    """
    Evaluates multiple strategies and computes a weighted confluence score.
    """
    
    def __init__(self, strategies: List[BaseStrategy]):
        self.strategies = strategies

    def evaluate_all(self, df: pd.DataFrame) -> Signal:
        """
        Run all strategies and compute a combined weighted signal.
        """
        if df.empty:
            return Signal(direction=0, score=0.0, reason="Empty dataframe", invalidation="")
            
        regime = detect_regime(df)
        
        buy_score = 0.0
        sell_score = 0.0
        buy_weight = 0.0
        sell_weight = 0.0
        reasons = []
        abstain_reasons = []
        
        for strategy in self.strategies:
            signal = strategy.evaluate(df, regime)
            weight = strategy.weight
            
            if signal.direction == 1:
                buy_score += signal.score * weight
                buy_weight += weight
                if signal.score > 50:
                    reasons.append(f"{strategy.name}: BUY ({signal.score:.1f})")
            elif signal.direction == -1:
                sell_score += signal.score * weight
                sell_weight += weight
                if signal.score > 50:
                    reasons.append(f"{strategy.name}: SELL ({signal.score:.1f})")
            # Neutral strategies abstain. They are NOT counted in the
            # denominator: doing so treated "no opinion" as "opinion against",
            # so a single unanimous 90/100 signal was diluted to 30/100 and
            # discarded whenever the other members abstained.
            else:
                abstain_reasons.append(f"{strategy.name}: neutral ({signal.reason})")

        if buy_weight == 0 and sell_weight == 0:
            # Nobody took a side. Report an honest zero rather than a residual
            # score: a neutral signal must never carry a tradable score.
            reason = "All strategies abstained"
            if abstain_reasons:
                reason += " (" + "; ".join(abstain_reasons) + ")"
            return Signal(direction=0, score=0.0, reason=reason, invalidation="")
            
        # Normalize each side by the weight that actually voted for it, so
        # abstentions cannot dilute a confident, unanimous signal.
        buy_score = buy_score / buy_weight if buy_weight else 0.0
        sell_score = sell_score / sell_weight if sell_weight else 0.0
        
        # Decide direction
        if buy_score > sell_score and buy_score > 55:
            return Signal(
                direction=1,
                score=buy_score,
                reason="; ".join(reasons),
                invalidation="Price drops below recent swing low/Fib invalidation or momentum flips bearish."
            )
        elif sell_score > buy_score and sell_score > 55:
            return Signal(
                direction=-1,
                score=sell_score,
                reason="; ".join(reasons),
                invalidation="Price breaks above recent swing high/Fib invalidation or momentum flips bullish."
            )
            
        return Signal(direction=0, score=0.0, reason="Insufficient confluence", invalidation="")


def get_default_ensemble() -> StrategyEnsemble:
    strategies = [
        TrendFollowingStrategy(weight=1.0),
        MomentumStrategy(weight=1.0),
        MeanReversionStrategy(weight=0.7),
        FibonacciStrategy(weight=1.3),
    ]
    return StrategyEnsemble(strategies)
