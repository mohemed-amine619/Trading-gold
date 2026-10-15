"""
core/strategy/strategies.py — Concrete Strategy Implementations.
"""

import math

import pandas as pd
from core.strategy.base import BaseStrategy, MarketRegime, Signal


def _num(row, column: str, default: float) -> float:
    """
    Read an indicator off a row, tolerating a missing *value*.

    ``pandas.Series.get(column, default)`` only substitutes when the column is
    absent. When the column exists but holds ``None`` — which is what
    ``pandas_ta`` emits during the warm-up bars of a long indicator such as
    ``ema200`` — ``.get`` returns that ``None`` unchanged, and a chained
    comparison such as ``close > ema50 > ema200`` raises ``TypeError`` on it.

    That made every symbol with less than 200 bars of history crash the whole
    ensemble rather than abstain: a fresh symbol, or any timeframe slower than
    the data available, could not be evaluated at all. An indicator that is not
    ready yet is *absent*, and absence must fall back, not explode.
    """
    value = row.get(column, default)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number

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
        ema50 = _num(latest, "ema50", close)
        ema200 = _num(latest, "ema200", close)
        adx = _num(latest, "adx", 0.0)
        di_plus = _num(latest, "di_plus", 0.0)
        di_minus = _num(latest, "di_minus", 0.0)

        # Bullish: price > ema50 > ema200
        if close > ema50 > ema200:
            score = 70.0
            if adx > 25 and di_plus > di_minus:
                score += 20.0
            return Signal(direction=1, score=score, reason="Uptrend (Price > EMA50 > EMA200)", invalidation="Price closes below EMA50")

        # Bearish: price < ema50 < ema200
        elif close < ema50 < ema200:
            score = 70.0
            if adx > 25 and di_minus > di_plus:
                score += 20.0
            return Signal(direction=-1, score=score, reason="Downtrend (Price < EMA50 < EMA200)", invalidation="Price closes above EMA50")
            
        return Signal(direction=0, score=0.0, reason="No clear trend", invalidation="")

class MomentumStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0):
        super().__init__(name="Momentum", weight=weight)
        
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        latest = df.iloc[-1]
        
        rsi = _num(latest, "rsi", 50.0)
        macd_hist = _num(latest, "macd_hist", 0.0)
        macd_hist_prev = _num(latest, "macd_hist_prev", 0.0)
        
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
        rsi = _num(latest, "rsi", 50.0)

        # Bullish reversal
        if rsi < 30:
            return Signal(direction=1, score=75.0, reason="Oversold (RSI < 30)", invalidation="RSI drops further below 20")
            
        # Bearish reversal
        elif rsi > 70:
            return Signal(direction=-1, score=75.0, reason="Overbought (RSI > 70)", invalidation="RSI climbs further above 80")
            
        return Signal(direction=0, score=0.0, reason="Price near mean", invalidation="")
