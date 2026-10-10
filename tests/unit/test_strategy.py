import pytest
import pandas as pd
from core.strategy.base import detect_regime, MarketRegime
from core.strategy.strategies import TrendFollowingStrategy, MomentumStrategy, MeanReversionStrategy
from core.strategy.ensemble import StrategyEnsemble

@pytest.fixture
def sample_df():
    # Create a dummy DataFrame with indicators
    return pd.DataFrame({
        "close": [100, 101, 102],
        "ema50": [90, 91, 92],
        "ema200": [80, 81, 82],
        "adx": [20, 22, 30],
        "di_plus": [25, 26, 28],
        "di_minus": [15, 14, 12],
        "rsi": [50, 55, 60],
        "macd_hist": [0.1, 0.2, 0.3],
        "macd_hist_prev": [0.0, 0.1, 0.2],
        "atr": [1.0, 1.1, 1.2]
    })

def test_detect_regime(sample_df):
    regime = detect_regime(sample_df)
    assert regime == MarketRegime.TRENDING

def test_trend_following_strategy(sample_df):
    strategy = TrendFollowingStrategy()
    signal = strategy.evaluate(sample_df, MarketRegime.TRENDING)
    
    assert signal.direction == 1
    assert signal.score > 70
    assert "Uptrend" in signal.reason

def test_momentum_strategy(sample_df):
    strategy = MomentumStrategy()
    signal = strategy.evaluate(sample_df, MarketRegime.TRENDING)
    
    assert signal.direction == 1
    assert signal.score == 80.0
    assert "Bullish MACD" in signal.reason

def test_mean_reversion_strategy(sample_df):
    strategy = MeanReversionStrategy()
    
    # In trending regime, it should return neutral
    signal1 = strategy.evaluate(sample_df, MarketRegime.TRENDING)
    assert signal1.direction == 0
    
    # In ranging, with rsi 60, it should return neutral
    signal2 = strategy.evaluate(sample_df, MarketRegime.RANGING)
    assert signal2.direction == 0
    
    # Modify RSI to be overbought
    df_overbought = sample_df.copy()
    df_overbought.loc[2, "rsi"] = 75
    
    signal3 = strategy.evaluate(df_overbought, MarketRegime.RANGING)
    assert signal3.direction == -1

def test_strategy_ensemble(sample_df):
    strategies = [
        TrendFollowingStrategy(weight=1.0),
        MomentumStrategy(weight=1.0),
        MeanReversionStrategy(weight=0.5)
    ]
    ensemble = StrategyEnsemble(strategies)
    
    signal = ensemble.evaluate_all(sample_df)
    
    # Both Trend and Momentum agree on BUY
    assert signal.direction == 1
    assert signal.score > 50
    assert "Trend: BUY" in signal.reason
    assert "Momentum: BUY" in signal.reason
