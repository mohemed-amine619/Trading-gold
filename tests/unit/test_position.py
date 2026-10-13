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
    original_sl = pos.sl
    
    # Entry: 2000, SL: 1990 => 1R risk = 10.
    # Pin the MARKET price to reach +1R, which activates trailing. A real
    # terminal reports one price in both get_tick() and position price_current;
    # the trailing logic validates the proposed stop against the live tick, so
    # the tick is what must move. 2012 puts BID at ~2011.9, comfortably past
    # the 10-point activation, without relying on an exact fill price.
    mock_adapter.set_price("XAUUSDm", 2012.0)
    
    manager.manage_positions(signal, pd.DataFrame())
    
    # Trailing distance is 0.5R = 5, measured from the live bid.
    bid = mock_adapter.get_tick("XAUUSDm").bid
    expected = bid - (abs(pos.price_open - original_sl) * manager.trail_distance_rr)
    assert mock_adapter._positions[0].sl == pytest.approx(expected)
    # The invariant that actually matters: the stop moved up and is never
    # wider than it was.
    assert mock_adapter._positions[0].sl > original_sl
    assert mock_adapter._positions[0].sl < bid


def test_position_breakeven(manager, mock_adapter):
    signal = Signal(direction=0, score=0.0, reason="", invalidation="")
    pos = mock_adapter._positions[0]
    original_sl = pos.sl
    
    # Entry: 2000, SL: 1990 => 1R risk = 10, breakeven fires at 0.5R = +5.
    # 2007 puts BID at ~2006.9, i.e. +6.9 — past the trigger with margin for
    # the half-spread between the pinned mid and the bid.
    mock_adapter.set_price("XAUUSDm", 2007.0)
    
    manager.manage_positions(signal, pd.DataFrame())
    
    # Breakeven moves the stop to entry + 2 ticks (tick_size 0.01 for gold),
    # which is a small buffer above entry to cover spread/commission.
    expected = pos.price_open + 2 * 0.01
    assert mock_adapter._positions[0].sl == pytest.approx(expected)
    # Breakeven must never be worse than the original stop.
    assert mock_adapter._positions[0].sl > original_sl
