"""
tests/unit/test_dynamic_sltp.py — Tests for Dynamic SL/TP and Local/Embedded AI Reasoning.
"""

import pytest
import pandas as pd
import numpy as np

from core.ai.dynamic_sltp import (
    SLTPSafetyGuard,
    DynamicSLTPReasoner,
    EntrySLTPPlan,
    PositionAdjustmentPlan,
)
from core.ai.provider import (
    EmbeddedReasoningProvider,
    LocalEndpointProvider,
    get_llm_provider,
)
from core.ai.decision import AIEngine
from core.strategy.base import Signal


# ─────────────────────────────────────────────────────────────────────────────
# 1. Safety Guardrail Tests (Never Loosen SL & Grid Snapping)
# ─────────────────────────────────────────────────────────────────────────────

def test_safety_guard_never_loosens_buy_sl():
    """For BUY: any new SL lower than or equal to current SL must be rejected."""
    guard = SLTPSafetyGuard()
    current_price = 2650.0
    current_sl = 2640.0
    tick_size = 0.01
    atr = 5.0

    # Tighter SL -> Approved
    approved = guard.validate_new_sl(
        direction=1,
        current_price=current_price,
        current_sl=current_sl,
        proposed_sl=2645.0,
        tick_size=tick_size,
        stops_level_pts=10,
        point=0.01,
        atr=atr,
    )
    assert approved == 2645.0

    # Looser SL (lower than current) -> REJECTED (None)
    loosened = guard.validate_new_sl(
        direction=1,
        current_price=current_price,
        current_sl=current_sl,
        proposed_sl=2635.0,
        tick_size=tick_size,
        stops_level_pts=10,
        point=0.01,
        atr=atr,
    )
    assert loosened is None

    # Equal SL -> REJECTED (None)
    equal_sl = guard.validate_new_sl(
        direction=1,
        current_price=current_price,
        current_sl=current_sl,
        proposed_sl=2640.0,
        tick_size=tick_size,
        stops_level_pts=10,
        point=0.01,
        atr=atr,
    )
    assert equal_sl is None


def test_safety_guard_never_loosens_sell_sl():
    """For SELL: any new SL higher than or equal to current SL must be rejected."""
    guard = SLTPSafetyGuard()
    current_price = 2650.0
    current_sl = 2660.0
    tick_size = 0.01
    atr = 5.0

    # Tighter SL (lower) -> Approved
    approved = guard.validate_new_sl(
        direction=-1,
        current_price=current_price,
        current_sl=current_sl,
        proposed_sl=2655.0,
        tick_size=tick_size,
        stops_level_pts=10,
        point=0.01,
        atr=atr,
    )
    assert approved == 2655.0

    # Looser SL (higher) -> REJECTED (None)
    loosened = guard.validate_new_sl(
        direction=-1,
        current_price=current_price,
        current_sl=current_sl,
        proposed_sl=2665.0,
        tick_size=tick_size,
        stops_level_pts=10,
        point=0.01,
        atr=atr,
    )
    assert loosened is None


def test_safety_guard_side_of_market():
    """BUY SL cannot be above current price; SELL SL cannot be below current price."""
    guard = SLTPSafetyGuard()
    current_price = 2650.0
    atr = 5.0

    # BUY SL above price -> rejected
    assert guard.validate_new_sl(
        direction=1,
        current_price=current_price,
        current_sl=2640.0,
        proposed_sl=2655.0,
        tick_size=0.01,
        stops_level_pts=5,
        point=0.01,
        atr=atr,
    ) is None

    # SELL SL below price -> rejected
    assert guard.validate_new_sl(
        direction=-1,
        current_price=current_price,
        current_sl=2660.0,
        proposed_sl=2645.0,
        tick_size=0.01,
        stops_level_pts=5,
        point=0.01,
        atr=atr,
    ) is None


def test_safety_guard_tick_snapping():
    """Prices must snap to broker tick grid half-up without binary float noise."""
    guard = SLTPSafetyGuard()
    # Forex 5 decimals: 0.0001 tick
    assert guard.snap_price(1.08556, 0.0001) == 1.0856
    assert guard.snap_price(1.08554, 0.0001) == 1.0855

    # Gold 2 decimals: 0.01 tick
    assert guard.snap_price(2650.1234, 0.01) == 2650.12
    assert guard.snap_price(2650.126, 0.01) == 2650.13


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dynamic SL/TP Reasoner Tests (Entries & Position Management)
# ─────────────────────────────────────────────────────────────────────────────

class MockSymbolInfo:
    def __init__(self, tick_size=0.01, point=0.01, stops_level=10, bid=2650.0, ask=2650.2):
        self.trade_tick_size = tick_size
        self.point = point
        self.trade_stops_level = stops_level
        self.bid = bid
        self.ask = ask
        self.spread = int(round((ask - bid) / point))


class MockPosition:
    def __init__(self, pos_type=0, price_open=2650.0, sl=2645.0, tp=2665.0, volume=0.1, symbol="XAUUSDm"):
        self.type = pos_type        # 0 = BUY, 1 = SELL
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.volume = volume
        self.symbol = symbol


def test_dynamic_sltp_entry_buy():
    """Dynamic entry SL/TP for BUY properly calculates risk, reward, and R:R."""
    reasoner = DynamicSLTPReasoner()
    symbol_info = MockSymbolInfo()
    entry_price = 2650.0
    atr = 4.0
    df = pd.DataFrame({"close": [2648.0, 2650.0], "high": [2651.0, 2652.0], "low": [2647.0, 2649.0]})

    plan = reasoner.calculate_entry_sltp(
        symbol="XAUUSDm",
        direction=1,
        entry_price=entry_price,
        df=df,
        atr=atr,
        symbol_info=symbol_info,
    )

    assert plan.valid is True
    assert plan.sl_price < entry_price
    assert plan.tp_price > entry_price
    assert plan.rr_ratio >= 1.5
    assert plan.sl_distance > 0
    assert plan.tp_distance > 0


def test_dynamic_sltp_entry_sell():
    """Dynamic entry SL/TP for SELL properly calculates risk, reward, and R:R."""
    reasoner = DynamicSLTPReasoner()
    symbol_info = MockSymbolInfo()
    entry_price = 2650.0
    atr = 4.0
    df = pd.DataFrame({"close": [2652.0, 2650.0], "high": [2653.0, 2651.0], "low": [2649.0, 2648.0]})

    plan = reasoner.calculate_entry_sltp(
        symbol="XAUUSDm",
        direction=-1,
        entry_price=entry_price,
        df=df,
        atr=atr,
        symbol_info=symbol_info,
    )

    assert plan.valid is True
    assert plan.sl_price > entry_price
    assert plan.tp_price < entry_price
    assert plan.rr_ratio >= 1.5


def test_dynamic_sltp_breakeven_ratchet():
    """When position profit hits the breakeven trigger, SL is ratcheted to entry + buffer."""
    reasoner = DynamicSLTPReasoner(config={"breakeven_trigger_atr": 0.40, "breakeven_buffer_atr": 0.15})
    # Price moved from 2650.0 to 2653.0 (+3.0 = 0.75 ATR profit on ATR=4.0)
    symbol_info = MockSymbolInfo(bid=2653.0, ask=2653.2)
    pos = MockPosition(pos_type=0, price_open=2650.0, sl=2645.0)
    df = pd.DataFrame({"close": [2650.0, 2653.0], "high": [2651.0, 2654.0], "low": [2649.0, 2650.0]})

    adjustment = reasoner.evaluate_position_sltp(
        position=pos,
        df=df,
        atr=4.0,
        symbol_info=symbol_info,
    )

    assert adjustment.action in ("MOVE_TO_BREAKEVEN", "TIGHTEN_SL")
    assert adjustment.new_sl is not None
    # Must be at or above breakeven (2650.0 + 0.15*4.0 = 2650.60)
    assert adjustment.new_sl >= 2650.60
    assert adjustment.new_sl > pos.sl


def test_dynamic_sltp_pinbar_profit_securing():
    """Opposing pinbar after profit causes dynamic SL to tighten immediately."""
    reasoner = DynamicSLTPReasoner()
    # Price in profit (+2.5 profit with ATR=4.0 is > 0.4 ATR)
    symbol_info = MockSymbolInfo(bid=2652.5, ask=2652.7)
    pos = MockPosition(pos_type=0, price_open=2650.0, sl=2645.0)
    # Opposing pinbar formed on the last bar
    df = pd.DataFrame({
        "close": [2651.0, 2652.5],
        "high": [2652.0, 2654.5],
        "low": [2650.0, 2651.0],
        "bear_pinbar": [False, True],
    })

    adjustment = reasoner.evaluate_position_sltp(
        position=pos,
        df=df,
        atr=4.0,
        symbol_info=symbol_info,
    )

    assert adjustment.action in ("SECURE_PROFIT", "TIGHTEN_SL")
    assert adjustment.new_sl is not None
    assert adjustment.new_sl > pos.sl


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedded AI Provider Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_embedded_ai_provider_direct():
    """Embedded provider correctly evaluates signals in pure Python without external servers."""
    provider = EmbeddedReasoningProvider()
    engine = AIEngine(provider)

    # Bullish confluence
    buy_signal = Signal(direction=1, score=75.0, reason="EMA cross + strong ADX", invalidation="")
    result = engine.evaluate_signal(buy_signal, "Market Context: Uptrend confirmed, ADX 30, discount zone.")
    assert result.direction == 1
    assert result.score > 0
    assert "Embedded AI confirms bullish" in result.reason

    # Bearish risk warning -> pass/rejection
    result_rejected = engine.evaluate_signal(buy_signal, "Market Context: Extreme resistance overhead, news incoming.")
    assert result_rejected.direction == 0
    assert result_rejected.score == 0.0


def test_local_endpoint_fallback():
    """LocalEndpointProvider falls back to EmbeddedReasoningProvider when server is unreachable."""
    # Port 59999 has no server running
    local_provider = LocalEndpointProvider(base_url="http://localhost:59999/v1", timeout=0.5)
    engine = AIEngine(local_provider)

    buy_signal = Signal(direction=1, score=70.0, reason="Tech Buy", invalidation="")
    result = engine.evaluate_signal(buy_signal, "Market context: Normal trading session.")

    # Must NOT crash, falls back safely to embedded AI
    assert result.direction == 1
    assert result.score > 0


def test_get_llm_provider_factory():
    """Factory correctly resolves embedded, local, and mock providers."""
    assert isinstance(get_llm_provider("embedded"), EmbeddedReasoningProvider)
    assert isinstance(get_llm_provider("none"), EmbeddedReasoningProvider)
    assert isinstance(get_llm_provider("local_endpoint"), LocalEndpointProvider)


def test_gold_volatility_stop_floor():
    """Gold trades enforce a minimum SL ATR multiple of 1.6 to prevent noise stop-outs."""
    reasoner = DynamicSLTPReasoner()
    symbol_info = MockSymbolInfo()
    entry_price = 2650.0
    atr = 4.0
    df = pd.DataFrame({"close": [2648.0, 2650.0], "high": [2649.0, 2651.0], "low": [2647.0, 2649.0]})

    plan = reasoner.calculate_entry_sltp(
        symbol="XAUUSDm",
        direction=1,
        entry_price=entry_price,
        df=df,
        atr=atr,
        symbol_info=symbol_info,
    )

    assert plan.valid is True
    # SL distance must be at least 1.6 * atr (1.6 * 4.0 = 6.4)
    assert plan.sl_distance >= 1.6 * atr
    assert "Gold volatility protection" in plan.reasoning


def test_gold_peak_profit_giveback_defense():
    """Gold scalps defend gains aggressively when profit pulls back from peak."""
    reasoner = DynamicSLTPReasoner()
    # Price reached peak 2655.0 (+5.0 = 1.25 ATR), now at 2653.0 (gave back 40%)
    symbol_info = MockSymbolInfo(bid=2653.0, ask=2653.2)
    pos = MockPosition(pos_type=0, price_open=2650.0, sl=2645.0, symbol="XAUUSDm")
    df = pd.DataFrame({"close": [2650.0, 2653.0], "high": [2651.0, 2655.0], "low": [2649.0, 2650.0]})

    adjustment = reasoner.evaluate_position_sltp(
        position=pos,
        df=df,
        atr=4.0,
        symbol_info=symbol_info,
        peak_profit=5.0,
    )

    assert adjustment.action in ("TIGHTEN_SL", "SECURE_PROFIT", "MOVE_TO_BREAKEVEN")
    assert adjustment.new_sl is not None
    assert adjustment.new_sl > pos.sl


# ─────────────────────────────────────────────────────────────────────────────
# 6. LLM Trader Council & OpenAICompatibleProvider Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_openai_compatible_provider_mock(monkeypatch):
    """OpenAICompatibleProvider formats requests and parses responses correctly."""
    from core.ai.provider import OpenAICompatibleProvider, get_llm_provider

    provider = get_llm_provider("openai_compatible", base_url="https://api.groq.com/openai/v1", model="llama-3.3-70b-versatile", api_key="test-key")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://api.groq.com/openai/v1"
    assert provider.model == "llama-3.3-70b-versatile"
    assert provider.api_key == "test-key"

    # Mock requests.post
    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"decision": "BUY", "confidence": 88.0, "reasoning": "Strong order flow", "risk_assessment": "Low"}'
                    }
                }]
            }

    import requests
    monkeypatch.setattr(requests, "post", lambda url, json, headers, timeout: MockResponse())

    res = provider.generate("Analyze setup", "You are quant", json_mode=True)
    assert '"decision": "BUY"' in res
    assert '"confidence": 88.0' in res


def test_openai_compatible_provider_fallback_on_network_error(monkeypatch):
    """When remote LLM times out or fails, fallback to embedded reasoning safely."""
    from core.ai.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(base_url="http://invalid-llm-host:1234/v1")

    import requests
    def mock_post_fail(*a, **k):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr(requests, "post", mock_post_fail)

    # Must NOT raise exception; returns embedded fallback
    res = provider.generate("Evaluate SL/TP", "System Prompt", json_mode=True)
    assert isinstance(res, str)
    assert "decision" in res


def test_evaluate_trade_candidate_critic_approval(monkeypatch):
    """DynamicSLTPReasoner.evaluate_trade_candidate approves high confluence setups."""
    from core.ai.dynamic_sltp import DynamicSLTPReasoner
    from core.ai.provider import MockLLMProvider

    class ConfirmingLLM(MockLLMProvider):
        def generate(self, prompt, system_prompt, json_mode=False):
            return '{"decision": "BUY", "confidence": 92.5, "reasoning": "Clear liquidity sweep at support", "risk_assessment": "Favorable 1:2.5 RR"}'

    reasoner = DynamicSLTPReasoner(provider=ConfirmingLLM())
    df = pd.DataFrame({
        "open": [2640.0, 2642.0], "high": [2645.0, 2646.0],
        "low": [2639.0, 2641.0], "close": [2642.0, 2645.0],
        "rsi": [55.0, 58.0], "adx": [24.0, 26.0], "atr": [4.0, 4.0]
    })

    critique = reasoner.evaluate_trade_candidate(
        symbol="XAUUSDm",
        direction=1,
        score=72.0,
        ai_conf=0.65,
        df=df,
    )

    assert critique.approved is True
    assert critique.decision == "BUY"
    assert critique.confidence == 92.5
    assert "liquidity sweep" in critique.reasoning


def test_evaluate_trade_candidate_critic_veto(monkeypatch):
    """DynamicSLTPReasoner.evaluate_trade_candidate vetoes setups when LLM critic detects trap."""
    from core.ai.dynamic_sltp import DynamicSLTPReasoner
    from core.ai.provider import MockLLMProvider

    class VetoingLLM(MockLLMProvider):
        def generate(self, prompt, system_prompt, json_mode=False):
            return '{"decision": "PASS", "confidence": 40.0, "reasoning": "Heavy overhead order block resistance, exhaustion detected", "risk_assessment": "High trap risk"}'

    reasoner = DynamicSLTPReasoner(provider=VetoingLLM())
    df = pd.DataFrame({
        "open": [2640.0, 2642.0], "high": [2645.0, 2646.0],
        "low": [2639.0, 2641.0], "close": [2642.0, 2645.0],
        "rsi": [78.0, 82.0], "adx": [15.0, 14.0], "atr": [4.0, 4.0]
    })

    critique = reasoner.evaluate_trade_candidate(
        symbol="XAUUSDm",
        direction=1,
        score=68.0,
        ai_conf=0.60,
        df=df,
    )

    assert critique.approved is False
    assert critique.decision == "PASS"
    assert "exhaustion detected" in critique.reasoning

