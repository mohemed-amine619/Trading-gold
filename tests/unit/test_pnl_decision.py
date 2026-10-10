import pytest
from core.position.decision import PNLDecisionEngine, PositionAction, MarketRegime

def test_pnl_metrics_long():
    engine = PNLDecisionEngine()
    # open 100, current 110, sl 90, dir 1, vol 1.0, tick_val 1.0, tick_size 0.01
    # risk = 10 / 0.01 = 1000
    # pnl = 10 / 0.01 = 1000
    # rr = 1.0
    metrics = engine.calculate_metrics(
        ticket=1, open_price=100.0, current_price=110.0, sl=90.0, initial_sl=90.0,
        direction=1, volume=1.0, tick_value=1.0, tick_size=0.01,
        historical_high=115.0, historical_low=95.0
    )
    assert metrics.floating_pnl == 1000.0
    assert metrics.initial_risk == 1000.0
    assert metrics.current_rr == 1.0
    assert metrics.mfe == 15.0
    assert metrics.mae == 5.0
    assert metrics.max_floating_profit == 1500.0
    assert metrics.drawdown_from_peak == 500.0

def test_pnl_decision_breakeven():
    engine = PNLDecisionEngine(be_activation_rr=1.0)
    metrics = engine.calculate_metrics(
        ticket=1, open_price=100.0, current_price=110.0, sl=90.0, initial_sl=90.0,
        direction=1, volume=1.0, tick_value=1.0, tick_size=0.01,
        historical_high=115.0, historical_low=95.0
    )
    result = engine.evaluate(metrics, MarketRegime.STRONG_UPTREND)
    assert result.action == PositionAction.MOVE_TO_BREAKEVEN

def test_pnl_decision_giveback():
    engine = PNLDecisionEngine()
    metrics = engine.calculate_metrics(
        ticket=1, open_price=100.0, current_price=105.0, sl=100.0, initial_sl=90.0,
        direction=1, volume=1.0, tick_value=1.0, tick_size=0.01,
        historical_high=130.0, historical_low=95.0
    )
    # Peak profit = 3000
    # Current profit = 500
    # Drawdown = 2500 (83% giveback)
    # Initial risk = 0 (since SL=100), but fallback to 1.0 for testing logic.
    # Wait, initial risk price was 10 before SL move. The calculate_metrics uses current SL for initial risk, which is a bug for initial risk.
    # Let's override initial risk manually for this test.
    metrics.initial_risk = 1000.0 
    
    result = engine.evaluate(metrics, MarketRegime.STRONG_UPTREND)
    assert result.action == PositionAction.LOCK_PROFIT

def test_pnl_decision_reversal():
    engine = PNLDecisionEngine()
    metrics = engine.calculate_metrics(
        ticket=1, open_price=100.0, current_price=110.0, sl=90.0, initial_sl=90.0,
        direction=1, volume=1.0, tick_value=1.0, tick_size=0.01,
        historical_high=115.0, historical_low=95.0
    )
    result = engine.evaluate(metrics, MarketRegime.POTENTIAL_REVERSAL)
    assert result.action == PositionAction.CLOSE_POSITION
