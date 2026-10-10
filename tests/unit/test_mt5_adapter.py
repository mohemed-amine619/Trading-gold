import pytest
import pandas as pd
from core.mt5.adapter import MockMT5Adapter
from core.mt5.types import Timeframe, OrderRequest, TradeAction, RetCode

@pytest.fixture
def mock_adapter():
    adapter = MockMT5Adapter(balance=10000.0, seed=42)
    adapter.connect()
    return adapter

def test_mock_connection(mock_adapter):
    assert mock_adapter.is_connected()
    
    account_info = mock_adapter.get_account_info()
    assert account_info is not None
    assert account_info.balance == 10000.0
    
    mock_adapter.disconnect()
    assert not mock_adapter.is_connected()
    assert mock_adapter.get_account_info() is None

def test_mock_symbol_info(mock_adapter):
    info = mock_adapter.get_symbol_info("XAUUSDm")
    assert info is not None
    assert info.name == "XAUUSDm"
    assert info.digits == 5
    
def test_mock_get_tick(mock_adapter):
    tick = mock_adapter.get_tick("XAUUSDm")
    assert tick is not None
    assert tick.symbol == "XAUUSDm"
    assert tick.bid > 0
    assert tick.ask > tick.bid

def test_mock_get_candles(mock_adapter):
    # Fetch 10 candles
    df = mock_adapter.get_candles("XAUUSDm", Timeframe.M15, 10, drop_forming=True)
    assert not df.empty
    assert len(df) == 10
    assert list(df.columns) == ["open", "high", "low", "close", "tick_volume", "spread"]
    
    # Values should make sense (random walk)
    assert df["open"].iloc[0] > 0
    assert df["high"].iloc[0] >= df["open"].iloc[0]

def test_mock_send_order(mock_adapter):
    request = OrderRequest(
        symbol="XAUUSDm",
        action=TradeAction.DEAL,
        type=0, # BUY
        volume=0.1,
        price=2000.0, # Not actually used in market fill for mock, it uses ask price
        sl=1990.0,
        tp=2020.0,
        magic=123,
        comment="test_order"
    )
    
    result = mock_adapter.send_order(request)
    assert result.retcode == RetCode.DONE
    assert result.deal > 0
    
    positions = mock_adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "XAUUSDm"
    assert positions[0].volume == 0.1
    assert positions[0].sl == 1990.0
    assert positions[0].tp == 2020.0
    assert positions[0].magic == 123
    assert positions[0].is_buy is True

def test_mock_calc_margin(mock_adapter):
    # order_type, symbol, volume, price
    margin = mock_adapter.calc_margin(0, "XAUUSDm", 1.0, 2000.0)
    # 1.0 volume * 100000 contract_size * 2000.0 / 100 leverage = 2,000,000
    assert margin == 2000000.0
