import pytest
import pandas as pd
from core.mt5.adapter import MockMT5Adapter
from core.execution.engine import ExecutionEngine
from core.position.manager import PositionManager
from core.strategy.base import Signal
from core.mt5.types import MT5Position

@pytest.fixture
def mock_adapter():
    adapter = MockMT5Adapter(balance=10000.0, seed=42)
    adapter.connect()
    # Inject a position manually for testing
    adapter._positions = [
        MT5Position(
            ticket=123,
            symbol="XAUUSDm",
            type=0, # BUY
            volume=0.1,
            price_open=2000.0,
            price_current=2010.0, # +$10 profit
            sl=1990.0, # $10 initial risk (1R)
            tp=2020.0, # $20 initial target (2R)
            magic=888999,
            time=1234567890
        )
    ]
    return adapter

@pytest.fixture
def manager(mock_adapter):
    execution = ExecutionEngine(adapter=mock_adapter)
    return PositionManager(adapter=mock_adapter, execution_engine=execution)

def test_position_thesis_invalidation(manager, mock_adapter):
    # Current signal is strong SELL (opposite of our BUY position)
    opposite_signal = Signal(direction=-1, score=80.0, reason="Strong downtrend", invalidation="")
    
    # Run manager
    manager.manage_positions(opposite_signal, pd.DataFrame())
    
    # Position should be closed (we mock closing by executing an opposite order)
    # The mock adapter just adds the opposite order to positions
    positions = mock_adapter.get_positions()
    assert len(positions) == 2
    assert positions[1].type == 1 # SELL to close
    assert positions[1].comment == "close_pos"

def test_position_trailing_stop(manager, mock_adapter):
    # Current signal is neutral
    signal = Signal(direction=0, score=0.0, reason="", invalidation="")
    
    pos = mock_adapter._positions[0]
    
    # Setup profit to hit trailing stop (1R profit)
    # Entry: 2000, SL: 1990 (Risk = 10)
    # Price = 2010 (Profit = 10 -> 1R)
    # Trail distance = 0.5R = 5
    # New SL = 2010 - 5 = 2005
    pos.price_current = 2010.0
    
    manager.manage_positions(signal, pd.DataFrame())
    
    # In mock, modify_position just updates the position in the list
    assert mock_adapter._positions[0].sl == pytest.approx(2005.0)

def test_position_breakeven(manager, mock_adapter):
    signal = Signal(direction=0, score=0.0, reason="", invalidation="")
    pos = mock_adapter._positions[0]
    
    # Entry: 2000, SL: 1990 (Risk = 10)
    # Breakeven activates at 0.5R = 5 profit
    pos.price_current = 2005.0
    
    manager.manage_positions(signal, pd.DataFrame())
    
    # New SL should be entry price + offset (0.00002)
    # SL = 2000.00002
    assert mock_adapter._positions[0].sl == pytest.approx(2000.00002)
