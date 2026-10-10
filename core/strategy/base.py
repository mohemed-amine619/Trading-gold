"""
core/strategy/base.py — Strategy Engine Base and Interfaces.

Defines the BaseStrategy interface and the MarketRegime detector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    UNCERTAIN = "UNCERTAIN"

@dataclass
class Signal:
    direction: int      # 1 for BUY, -1 for SELL, 0 for NEUTRAL
    score: float        # 0.0 to 100.0
    reason: str         # Human-readable explanation
    invalidation: str   # Condition that invalidates the thesis

class BaseStrategy(ABC):
    """
    Abstract base class for all quantitative strategies.
    Strategies process OHLCV + Indicators to produce a Signal.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        """
        Evaluate market conditions and return a Signal.
        """
        ...
        
def detect_regime(df: pd.DataFrame) -> MarketRegime:
    """
    Detect the current market regime using basic indicators.
    Assumes df has indicators from the pipeline.
    """
    if df.empty or "adx" not in df.columns:
        return MarketRegime.UNCERTAIN
        
    latest = df.iloc[-1]
    
    adx = latest.get("adx", 0.0)
    atr = latest.get("atr", 0.0)
    
    # Very basic regime detection logic
    if adx > 25:
        return MarketRegime.TRENDING
    elif adx < 20:
        if atr > df["atr"].rolling(20).mean().iloc[-1] * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        return MarketRegime.RANGING
        
    return MarketRegime.UNCERTAIN
