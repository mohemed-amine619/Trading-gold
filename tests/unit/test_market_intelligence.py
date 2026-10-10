import pytest
import pandas as pd
from core.market.intelligence import MarketIntelligenceEngine, MarketRegime

def test_market_intelligence_uptrend():
    engine = MarketIntelligenceEngine()
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104] * 6,
        'high': [101, 102, 103, 104, 105] * 6,
        'low': [99, 100, 101, 102, 103] * 6,
        'sma_fast': [105]*30,
        'sma_slow': [100]*30,
        'adx': [30]*30,
        'rsi': [65]*30,
        'atr': [2.0]*30
    })
    result = engine.classify_regime(df)
    assert result.regime == MarketRegime.STRONG_UPTREND
    assert result.confidence > 0.8

def test_market_intelligence_correction_down():
    engine = MarketIntelligenceEngine()
    df = pd.DataFrame({
        'close': [104]*30,
        'high': [101]*30,
        'low': [99]*30,
        'sma_fast': [105]*30,
        'sma_slow': [100]*30,
        'adx': [30]*30,
        'rsi': [35]*30,  # RSI low in an uptrend -> pullback
        'atr': [2.0]*30
    })
    result = engine.classify_regime(df)
    assert result.regime == MarketRegime.CORRECTION_DOWN

def test_market_intelligence_reversal():
    engine = MarketIntelligenceEngine()
    df = pd.DataFrame({
        'close': [100.5]*30,
        'high': [101]*30,
        'low': [99]*30,
        'sma_fast': [105]*30,
        'sma_slow': [100]*30, # distance to slow is 0.5 < atr(2.0)
        'adx': [30]*30,
        'rsi': [35]*30,
        'atr': [2.0]*30
    })
    result = engine.classify_regime(df)
    assert result.regime == MarketRegime.POTENTIAL_REVERSAL

def test_market_intelligence_ranging():
    engine = MarketIntelligenceEngine()
    df = pd.DataFrame({
        'close': [104]*30,
        'high': [101]*30,
        'low': [99]*30,
        'sma_fast': [100]*30,
        'sma_slow': [100]*30,
        'adx': [15]*30,
        'rsi': [50]*30,
        'atr': [2.0]*30
    })
    result = engine.classify_regime(df)
    assert result.regime == MarketRegime.RANGING
