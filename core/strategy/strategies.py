"""
core/strategy/strategies.py — Concrete Strategy Implementations.
"""

import pandas as pd
from core.strategy.base import BaseStrategy, MarketRegime, Signal

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="Trend", weight=weight)
        
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        # Require trending regime for strong signals
        if regime not in (MarketRegime.TRENDING, MarketRegime.BREAKOUT):
            return Signal(direction=0, score=0.0, reason="Not trending regime", invalidation="")
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        close = latest["close"]
        ema50 = latest.get("ema50", close)
        ema200 = latest.get("ema200", close)
        
        # Bullish: price > ema50 > ema200
        if close > ema50 > ema200:
            score = 70.0
            if latest.get("adx", 0) > 25 and latest.get("di_plus", 0) > latest.get("di_minus", 0):
                score += 20.0
            return Signal(direction=1, score=score, reason="Uptrend (Price > EMA50 > EMA200)", invalidation="Price closes below EMA50")
            
        # Bearish: price < ema50 < ema200
        elif close < ema50 < ema200:
            score = 70.0
            if latest.get("adx", 0) > 25 and latest.get("di_minus", 0) > latest.get("di_plus", 0):
                score += 20.0
            return Signal(direction=-1, score=score, reason="Downtrend (Price < EMA50 < EMA200)", invalidation="Price closes above EMA50")
            
        return Signal(direction=0, score=0.0, reason="No clear trend", invalidation="")

class MomentumStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="Momentum", weight=weight)
        
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        latest = df.iloc[-1]
        
        rsi = latest.get("rsi", 50.0)
        macd_hist = latest.get("macd_hist", 0.0)
        macd_hist_prev = latest.get("macd_hist_prev", 0.0)
        
        # Bullish momentum
        if macd_hist > 0 and macd_hist > macd_hist_prev and 40 < rsi < 70:
            return Signal(direction=1, score=80.0, reason="Bullish MACD Expansion", invalidation="MACD Histogram ticks down")
            
        # Bearish momentum
        if macd_hist < 0 and macd_hist < macd_hist_prev and 30 < rsi < 60:
            return Signal(direction=-1, score=80.0, reason="Bearish MACD Expansion", invalidation="MACD Histogram ticks up")
            
        return Signal(direction=0, score=0.0, reason="Neutral Momentum", invalidation="")

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="MeanReversion", weight=weight)
        
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        if regime == MarketRegime.TRENDING:
            return Signal(direction=0, score=0.0, reason="Trending regime, mean reversion disabled", invalidation="")
            
        latest = df.iloc[-1]
        rsi = latest.get("rsi", 50.0)
        
        # Bullish reversal
        if rsi < 30:
            return Signal(direction=1, score=75.0, reason="Oversold (RSI < 30)", invalidation="RSI drops further below 20")
            
        # Bearish reversal
        elif rsi > 70:
            return Signal(direction=-1, score=75.0, reason="Overbought (RSI > 70)", invalidation="RSI climbs further above 80")
            
        return Signal(direction=0, score=0.0, reason="Price near mean", invalidation="")
