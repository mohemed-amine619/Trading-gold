import pytest
from core.mt5.adapter import MockMT5Adapter
from core.execution.engine import ExecutionEngine
from core.risk.engine import TradeProposal
from core.mt5.types import RetCode

@pytest.fixture
def mock_adapter():
    adapter = MockMT5Adapter(balance=10000.0, seed=42)
    adapter.connect()
    return adapter

@pytest.fixture
def execution_engine(mock_adapter):
    return ExecutionEngine(adapter=mock_adapter, max_retries=3, retry_delay_ms=10)

def test_execute_proposal_success(execution_engine, mock_adapter):
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        strategy_name="test_strat"
    )
    
    result = execution_engine.execute_proposal(proposal, lot_size=0.1)
    
    assert result is not None
    assert result.success is True
    assert result.deal > 0
    
    positions = mock_adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].volume == 0.1
    assert positions[0].sl == 1990.0

def test_execute_proposal_zero_lots(execution_engine):
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        strategy_name="test_strat"
    )
    
    result = execution_engine.execute_proposal(proposal, lot_size=0.0)
    assert result is None

def test_modify_position(execution_engine, mock_adapter):
    # Open position first
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        strategy_name="test_strat"
    )
    res1 = execution_engine.execute_proposal(proposal, lot_size=0.1)
    
    ticket = res1.deal
    
    # Modify position
    res2 = execution_engine.modify_position(ticket, "XAUUSDm", new_sl=1995.0, new_tp=2025.0)
    assert res2.success is True
    
    positions = mock_adapter.get_positions()
    assert positions[0].sl == 1995.0
    assert positions[0].tp == 2025.0

class RequoteMockAdapter(MockMT5Adapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0
        
    def send_order(self, request):
        self.call_count += 1
        if self.call_count < 3:
            # First two times, simulate requote
            from core.mt5.types import OrderResult
            return OrderResult(retcode=RetCode.REQUOTE)
        return super().send_order(request)

def test_execute_retry_logic():
    adapter = RequoteMockAdapter(balance=10000.0)
    adapter.connect()
    
    engine = ExecutionEngine(adapter=adapter, max_retries=3, retry_delay_ms=10)
    
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        strategy_name="test_strat"
    )
    
    result = engine.execute_proposal(proposal, lot_size=0.1)
    
    assert result.success is True
    assert adapter.call_count == 3
