import math
from enum import Enum
from typing import Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

class MarketRegime(str, Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    CORRECTION_UP = "CORRECTION_UP"
    CORRECTION_DOWN = "CORRECTION_DOWN"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    POTENTIAL_REVERSAL = "POTENTIAL_REVERSAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNCERTAIN = "UNCERTAIN"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"

@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    reasons: list[str]
    volatility_ratio: float
    trend_strength: float

class MarketIntelligenceEngine:
    def __init__(self, atr_period: int = 14, rsi_period: int = 14):
        self.atr_period = atr_period
        self.rsi_period = rsi_period

    def classify_regime(self, df: pd.DataFrame) -> RegimeResult:
        if len(df) < self.atr_period * 2:
            return RegimeResult(MarketRegime.UNCERTAIN, 0.0, ["Insufficient data"], 1.0, 0.0)

        # Assuming df has 'close', 'high', 'low', 'atr', 'rsi', 'sma_fast', 'sma_slow', 'adx'
        # If not, we calculate basic metrics
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        
        # Trend indicators
        sma_fast = df['sma_fast'].iloc[-1] if 'sma_fast' in df else df['close'].rolling(10).mean().iloc[-1]
        sma_slow = df['sma_slow'].iloc[-1] if 'sma_slow' in df else df['close'].rolling(50).mean().iloc[-1]
        
        adx = df['adx'].iloc[-1] if 'adx' in df else 20.0  # Default to 20 if missing
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50.0
        atr = df['atr'].iloc[-1] if 'atr' in df else (df['high'] - df['low']).mean()
        
        reasons = []
        confidence = 0.5
        
        # Volatility check
        recent_tr = max(df['high'].iloc[-1] - df['low'].iloc[-1], abs(df['high'].iloc[-1] - close_prev), abs(df['low'].iloc[-1] - close_prev))
        volatility_ratio = recent_tr / atr if atr > 0 else 1.0
        
        if volatility_ratio > 3.0:
            reasons.append(f"Extreme volatility spike: {volatility_ratio:.1f}x ATR")
            return RegimeResult(MarketRegime.HIGH_VOLATILITY, 0.8, reasons, volatility_ratio, adx)

        # Basic trend determination
        is_uptrend = sma_fast > sma_slow
        is_downtrend = sma_fast < sma_slow
        
        if adx > 25:
            if is_uptrend:
                if rsi > 60:
                    regime = MarketRegime.STRONG_UPTREND
                    reasons.append("ADX > 25, Fast MA > Slow MA, RSI > 60")
                    confidence = 0.9
                elif rsi < 40:
                    regime = MarketRegime.CORRECTION_DOWN
                    reasons.append("ADX > 25 (Uptrend) but RSI < 40 (Pullback)")
                    confidence = 0.7
                else:
                    regime = MarketRegime.STRONG_UPTREND
                    confidence = 0.6
            elif is_downtrend:
                if rsi < 40:
                    regime = MarketRegime.STRONG_DOWNTREND
                    reasons.append("ADX > 25, Fast MA < Slow MA, RSI < 40")
                    confidence = 0.9
                elif rsi > 60:
                    regime = MarketRegime.CORRECTION_UP
                    reasons.append("ADX > 25 (Downtrend) but RSI > 60 (Pullback)")
                    confidence = 0.7
                else:
                    regime = MarketRegime.STRONG_DOWNTREND
                    confidence = 0.6
            else:
                regime = MarketRegime.UNCERTAIN
        else:
            if 40 < rsi < 60:
                regime = MarketRegime.RANGING
                reasons.append("ADX < 25 and RSI neutral")
                confidence = 0.8
            else:
                regime = MarketRegime.RANGING
                confidence = 0.5
                
        # Reversal detection (Correction vs Reversal)
        if regime in [MarketRegime.CORRECTION_UP, MarketRegime.CORRECTION_DOWN]:
            # Evaluate depth
            distance_to_slow = abs(close - sma_slow)
            if distance_to_slow < atr:
                regime = MarketRegime.POTENTIAL_REVERSAL
                reasons.append("Price crossing critical slow moving average during correction")
                confidence = 0.6

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            reasons=reasons,
            volatility_ratio=volatility_ratio,
            trend_strength=adx
        )
