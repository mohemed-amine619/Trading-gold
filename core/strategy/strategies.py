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

        # It reads the previous bar, so one bar of history is not enough. The
        # backtest engine asks for a signal on a one-bar slice; abstain there
        # rather than crash the whole run.
        if len(df) < 2:
            return Signal(direction=0, score=0.0, reason="Need two bars", invalidation="")

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


class FibonacciStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.4):
        super().__init__(name="Fibonacci", weight=weight)
        
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        if len(df) < 60:
            return Signal(direction=0, score=0.0, reason="Insufficient data for Fibonacci", invalidation="")
        
        # Find recent swing high and low (more recent swing)
        recent = df.tail(60)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        swing_high_idx = recent['high'].idxmax()
        swing_low_idx = recent['low'].idxmin()
        
        # Prefer more recent swing
        if swing_high_idx < swing_low_idx and (recent.index[-1] - swing_low_idx).days < 2:
            pass
        elif swing_low_idx < swing_high_idx and (recent.index[-1] - swing_high_idx).days < 2:
            pass
        
        if swing_high <= swing_low:
            return Signal(direction=0, score=0.0, reason="Invalid swing range", invalidation="")
        
        range_size = swing_high - swing_low
        # Fibonacci retracement levels
        fib_236 = swing_high - range_size * 0.236
        fib_382 = swing_high - range_size * 0.382
        fib_500 = swing_high - range_size * 0.5
        fib_618 = swing_high - range_size * 0.618
        fib_786 = swing_high - range_size * 0.786
        fib_886 = swing_high - range_size * 0.886
        
        latest = df.iloc[-1]
        close = latest['close']
        high = latest['high']
        low = latest['low']
        open_price = latest['open']
        
        # Check for bullish reversal at Fibonacci support levels (golden zone)
        # Price in discount zone bouncing off 61.8% or 50%
        if close <= fib_618 + range_size * 0.008 and close >= fib_786 - range_size * 0.008:
            if low <= fib_618 + range_size * 0.003:  # Wick tested the level
                score = 88.0
                if regime == MarketRegime.TRENDING:
                    score += 12.0
                elif regime == MarketRegime.RANGING:
                    score += 5.0
                # Bonus for bullish candle
                if close > open_price:
                    score += 3.0
                return Signal(direction=1, score=min(score, 98.0), 
                             reason=f"Fib 61.8% golden zone bounce (swing {swing_low:.4f}-{swing_high:.4f})", 
                             invalidation=f"Close below {fib_786:.4f}")
        
        # Also check 50% retracement
        if close <= fib_500 + range_size * 0.01 and close >= fib_618 - range_size * 0.01:
            if low <= fib_500 + range_size * 0.003:
                score = 85.0
                if regime == MarketRegime.TRENDING:
                    score += 10.0
                if close > open_price:
                    score += 3.0
                return Signal(direction=1, score=min(score, 95.0), 
                             reason=f"Fib 50% retracement bounce with confluence", 
                             invalidation=f"Close below {fib_618:.4f}")
        
        # Check for bearish reversal at Fibonacci resistance levels
        if close >= fib_382 - range_size * 0.008 and close <= fib_500 + range_size * 0.008:
            if high >= fib_382 - range_size * 0.003:  # Wick rejected
                score = 88.0
                if regime == MarketRegime.TRENDING:
                    score += 12.0
                elif regime == MarketRegime.RANGING:
                    score += 5.0
                if close < open_price:
                    score += 3.0
                return Signal(direction=-1, score=min(score, 98.0), 
                             reason=f"Fib 38.2-50% rejection zone", 
                             invalidation=f"Close above {fib_382:.4f}")
        
        # Check 23.6% in strong trends
        if regime == MarketRegime.TRENDING:
            if close >= fib_236 - range_size * 0.01 and close <= fib_382 + range_size * 0.01:
                if high >= fib_236 - range_size * 0.003 and close < open_price:
                    score = 83.0
                    return Signal(direction=-1, score=min(score, 92.0), 
                                 reason=f"Fib 23.6% rejection in downtrend context", 
                                 invalidation=f"Close above {fib_236:.4f}")
                if low <= fib_236 + range_size * 0.003 and close > open_price:
                    score = 83.0
                    return Signal(direction=1, score=min(score, 92.0), 
                                 reason=f"Fib 23.6% bounce in uptrend context", 
                                 invalidation=f"Close below {fib_236:.4f}")
        
        return Signal(direction=0, score=0.0, reason="No Fibonacci confluence", invalidation="")
