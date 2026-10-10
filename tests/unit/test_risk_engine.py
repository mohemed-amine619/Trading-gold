import pytest
import time
from datetime import date
from core.risk.engine import RiskEngine, TradeProposal, RiskVerdict
from core.mt5.types import AccountInfo, SymbolInfo, Tick, Timeframe
from datetime import datetime, timezone

@pytest.fixture
def risk_engine():
    # Use small loss streak cooldown for testing
    return RiskEngine(
        risk_per_trade=0.01,
        max_daily_loss_pct=0.02,
        max_weekly_loss_pct=0.05,
        max_drawdown_pct=0.10,
        max_open_positions=2,
        min_rr_ratio=1.5,
        max_margin_usage=0.50,
        max_spread_points=50.0,
        max_consecutive_losses=3,
        loss_streak_cooldown_minutes=1, # 1 minute
        dynamic_risk=False,
    )

@pytest.fixture
def mock_account():
    return AccountInfo(
        login=123,
        server="mock",
        balance=10000.0,
        equity=10000.0,
        margin=1000.0,
    )

@pytest.fixture
def mock_symbol():
    return SymbolInfo(
        name="XAUUSDm",
        point=0.01,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        volume_min=0.01,
        volume_step=0.01,
    )

@pytest.fixture
def mock_tick():
    return Tick(
        symbol="XAUUSDm",
        time=datetime.now(timezone.utc),
        bid=2000.0,
        ask=2000.2, # Spread is 0.2
    )

def test_risk_approve_valid_trade(risk_engine, mock_account, mock_symbol, mock_tick):
    risk_engine.update_equity(10000.0)
    
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=1990.2,  # $10 SL
        take_profit=2020.2, # $20 TP (RR=2)
        strategy_name="test",
        account_info=mock_account,
        symbol_info=mock_symbol,
        tick=mock_tick,
    )
    
    # 0 open positions, 1000 used margin
    result = risk_engine.approve(proposal, open_positions_count=0, adx=25.0, used_margin=1000.0)
    
    assert result.verdict == RiskVerdict.APPROVED
    assert result.approved is True
    # lot size: risk_amount = 10000 * 0.01 = 100
    # sl_distance = 10.0, sl_ticks = 10.0 / 0.01 = 1000
    # risk_per_lot = 1000 * 1.0 = 1000
    # raw_lots = 100 / 1000 = 0.1
    assert result.calculated_lot_size == 0.1

def test_risk_reject_missing_sl(risk_engine, mock_account):
    risk_engine.update_equity(10000.0)
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=0.0,
        take_profit=2020.2,
        strategy_name="test",
        account_info=mock_account,
    )
    
    result = risk_engine.approve(proposal, open_positions_count=0)
    assert result.verdict == RiskVerdict.REJECTED
    assert "stop_loss_required" in result.reason

def test_risk_reject_invalid_sl_direction(risk_engine, mock_account):
    risk_engine.update_equity(10000.0)
    # Buy trade, SL above entry
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=2010.2, 
        take_profit=2020.2,
        strategy_name="test",
        account_info=mock_account,
    )
    
    result = risk_engine.approve(proposal, open_positions_count=0)
    assert result.verdict == RiskVerdict.REJECTED
    assert "sl_invalid" in result.reason

def test_risk_reject_poor_rr(risk_engine, mock_account):
    risk_engine.update_equity(10000.0)
    # SL = 10, TP = 10, RR = 1.0, minimum is 1.5
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=1990.2,
        take_profit=2010.2,
        strategy_name="test",
        account_info=mock_account,
    )
    
    result = risk_engine.approve(proposal, open_positions_count=0)
    assert result.verdict == RiskVerdict.REJECTED
    assert "rr_ratio" in result.reason

def test_risk_reject_max_positions(risk_engine, mock_account, mock_symbol, mock_tick):
    risk_engine.update_equity(10000.0)
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=1990.2,
        take_profit=2020.2,
        strategy_name="test",
        account_info=mock_account,
        symbol_info=mock_symbol,
        tick=mock_tick,
    )
    # max_open_positions is 2
    result = risk_engine.approve(proposal, open_positions_count=2)
    assert result.verdict == RiskVerdict.REJECTED
    assert "max_positions" in result.reason

def test_risk_reject_daily_loss(risk_engine, mock_account, mock_symbol, mock_tick):
    risk_engine.update_equity(10000.0) # Start equity
    
    # Simulate loss dropping equity to 9750 (2.5% loss, max is 2.0%)
    mock_account.equity = 9750.0
    
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=1990.2,
        take_profit=2020.2,
        strategy_name="test",
        account_info=mock_account,
        symbol_info=mock_symbol,
        tick=mock_tick,
    )
    
    result = risk_engine.approve(proposal, open_positions_count=0)
    assert result.verdict == RiskVerdict.REJECTED
    assert "daily_loss_limit" in result.reason

def test_risk_reject_emergency_stop(risk_engine, mock_account, mock_symbol, mock_tick):
    risk_engine.update_equity(10000.0)
    risk_engine.trigger_emergency_stop()
    
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=1990.2,
        take_profit=2020.2,
        strategy_name="test",
        account_info=mock_account,
        symbol_info=mock_symbol,
        tick=mock_tick,
    )
    
    result = risk_engine.approve(proposal, open_positions_count=0)
    assert result.verdict == RiskVerdict.REJECTED
    assert "emergency stop" in result.reason

def test_consecutive_losses_cooldown(risk_engine, mock_account, mock_symbol, mock_tick):
    risk_engine.update_equity(10000.0)
    
    # Record 3 losses
    risk_engine.record_trade_result(-10.0)
    risk_engine.record_trade_result(-10.0)
    risk_engine.record_trade_result(-10.0)
    
    proposal = TradeProposal(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.2,
        stop_loss=1990.2,
        take_profit=2020.2,
        strategy_name="test",
        account_info=mock_account,
        symbol_info=mock_symbol,
        tick=mock_tick,
    )
    
    result = risk_engine.approve(proposal, open_positions_count=0)
    assert result.verdict == RiskVerdict.REJECTED
    assert "loss_streak_cooldown" in result.reason

    # We won't test time.sleep in a unit test effectively, but we can verify it's blocked.
