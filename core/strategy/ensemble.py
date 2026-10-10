"""
core/strategy/ensemble.py — Strategy Ensemble and Scoring.
"""

import logging
from typing import List
import pandas as pd

from core.strategy.base import BaseStrategy, MarketRegime, Signal, detect_regime

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
        total_weight = 0.0
        reasons = []
        
        for strategy in self.strategies:
            signal = strategy.evaluate(df, regime)
            weight = strategy.weight
            
            if signal.direction == 1:
                buy_score += signal.score * weight
                if signal.score > 50:
                    reasons.append(f"{strategy.name}: BUY ({signal.score:.1f})")
            elif signal.direction == -1:
                sell_score += signal.score * weight
                if signal.score > 50:
                    reasons.append(f"{strategy.name}: SELL ({signal.score:.1f})")
                    
            total_weight += weight

        if total_weight == 0:
            return Signal(direction=0, score=0.0, reason="No strategies active", invalidation="")
            
        # Normalize scores
        buy_score /= total_weight
        sell_score /= total_weight
        
        # Decide direction
        if buy_score > sell_score and buy_score > 50:
            return Signal(
                direction=1,
                score=buy_score,
                reason="; ".join(reasons),
                invalidation="Price drops below recent support or momentum flips bearish."
            )
        elif sell_score > buy_score and sell_score > 50:
            return Signal(
                direction=-1,
                score=sell_score,
                reason="; ".join(reasons),
                invalidation="Price breaks above recent resistance or momentum flips bullish."
            )
            
        return Signal(direction=0, score=max(buy_score, sell_score), reason="Insufficient confluence", invalidation="")
