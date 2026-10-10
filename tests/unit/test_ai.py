import pytest
import json
from core.ai.provider import MockLLMProvider
from core.ai.decision import AIEngine
from core.strategy.base import Signal

def test_ai_engine_agree():
    provider = MockLLMProvider()
    engine = AIEngine(provider)
    
    tech_signal = Signal(direction=1, score=70.0, reason="Tech Buy", invalidation="")
    
    final_signal = engine.evaluate_signal(tech_signal, "Market is bullish.")
    
    # Provider mock defaults to BUY with 85 confidence
    assert final_signal.direction == 1
    assert final_signal.score == (70.0 + 85.0) / 2.0
    assert "Tech Buy" in final_signal.reason
    assert "Strong bullish momentum" in final_signal.reason

def test_ai_engine_disagree():
    # Mock a SELL response
    sell_response = json.dumps({
        "decision": "SELL",
        "confidence": 90.0,
        "reasoning": "Bearish divergence.",
        "risk_assessment": "High risk."
    })
    
    provider = MockLLMProvider(responses={})
    # We override the generate method to always return the sell response
    provider.generate = lambda p, sp, json_mode=False: sell_response
    
    engine = AIEngine(provider)
    
    tech_signal = Signal(direction=1, score=70.0, reason="Tech Buy", invalidation="")
    
    final_signal = engine.evaluate_signal(tech_signal, "Market is bearish.")
    
    # Should neutralize to 0
    assert final_signal.direction == 0
    assert final_signal.score == 0.0
    assert "disagreement" in final_signal.reason

def test_ai_engine_pass():
    pass_response = json.dumps({
        "decision": "PASS",
        "confidence": 100.0,
        "reasoning": "Too risky due to news.",
        "risk_assessment": "Extreme risk."
    })
    
    provider = MockLLMProvider()
    provider.generate = lambda p, sp, json_mode=False: pass_response
    
    engine = AIEngine(provider)
    tech_signal = Signal(direction=1, score=70.0, reason="Tech Buy", invalidation="")
    
    final_signal = engine.evaluate_signal(tech_signal, "News incoming.")
    
    assert final_signal.direction == 0

def test_ai_engine_fallback_on_error():
    provider = MockLLMProvider()
    # Return invalid JSON
    provider.generate = lambda p, sp, json_mode=False: "This is not JSON"
    
    engine = AIEngine(provider)
    tech_signal = Signal(direction=1, score=70.0, reason="Tech Buy", invalidation="")
    
    final_signal = engine.evaluate_signal(tech_signal, "Market context")
    
    # Should fallback to technical signal with penalty
    assert final_signal.direction == 1
    assert final_signal.score == 70.0 * 0.9
    assert "unavailable/invalid" in final_signal.reason
