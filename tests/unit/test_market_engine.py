import pytest
import pandas as pd
from core.mt5.adapter import MockMT5Adapter
from core.mt5.types import Timeframe
from core.market.engine import MarketEngine

@pytest.fixture
def mock_adapter():
    adapter = MockMT5Adapter(balance=10000.0, seed=42)
    adapter.connect()
    return adapter

@pytest.fixture
def market_engine(mock_adapter):
    return MarketEngine(adapter=mock_adapter, symbols=["XAUUSDm"])

def test_market_engine_fetch(market_engine):
    # This should hit the adapter and the pipeline
    df = market_engine.get_data("XAUUSDm", Timeframe.M15, count=100)
    
    assert not df.empty
    assert len(df) == 100
    
    # Check that basic OHLCV columns exist
    assert "close" in df.columns
    
    # Check that indicators were added
    assert "ema50" in df.columns
    assert "rsi" in df.columns
    assert "atr" in df.columns

def test_market_engine_caching(market_engine):
    # First fetch
    df1 = market_engine.get_data("XAUUSDm", Timeframe.M15, count=100)
    
    # Second fetch should use cache (we can't easily mock time, but we can verify it doesn't fail and returns same data)
    df2 = market_engine.get_data("XAUUSDm", Timeframe.M15, count=100)
    
    pd.testing.assert_frame_equal(df1, df2)
    
def test_market_engine_tick(market_engine):
    tick = market_engine.get_tick("XAUUSDm")
    assert tick is not None
    assert tick.symbol == "XAUUSDm"
    assert tick.bid > 0
