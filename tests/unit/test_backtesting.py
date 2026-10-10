import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.backtesting.engine import BacktestEngine
from core.strategy.base import BaseStrategy, Signal, MarketRegime

class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="Dummy")
        self.call_count = 0
        
    def evaluate(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        self.call_count += 1
        # Buy on the 5th candle
        if self.call_count == 5:
            return Signal(direction=1, score=100.0, reason="Buy", invalidation="")
        return Signal(direction=0, score=0.0, reason="", invalidation="")

def generate_dummy_data():
    dates = [datetime(2023, 1, 1) + timedelta(minutes=15*i) for i in range(20)]
    
    # We want a buy on candle 5, then price goes up to hit TP
    prices = np.linspace(1000, 1020, 20)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices + 1,
        'atr': np.ones(20) * 2.0
    }, index=dates)
    
    return df

def test_backtest_engine_run():
    df = generate_dummy_data()
    strategy = DummyStrategy()
    
    engine = BacktestEngine(initial_capital=10000.0)
    metrics = engine.run(df, strategy)
    
    assert metrics["total_trades"] > 0
    assert metrics["final_capital"] != 10000.0
    
    # Check that equity curve was populated
    assert len(engine.equity_curve) == len(df) - 1
