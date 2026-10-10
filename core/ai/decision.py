"""
core/ai/decision.py — AI Decision Engine.

Coordinates the Market Analyst and Risk Critic roles to validate signals.
"""

import logging
import json
from pydantic import BaseModel, ValidationError

from core.ai.provider import LLMProvider
from core.strategy.base import Signal

logger = logging.getLogger(__name__)

class AIDecisionSchema(BaseModel):
    decision: str          # "BUY", "SELL", or "PASS"
    confidence: float      # 0.0 to 100.0
    reasoning: str
    risk_assessment: str

class AIEngine:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
    def evaluate_signal(self, technical_signal: Signal, market_context: str) -> Signal:
        """
        Takes a technical signal and asks the LLM to validate it.
        Returns a modified or confirmed Signal.
        """
        if technical_signal.direction == 0:
            return technical_signal
            
        direction_str = "BUY" if technical_signal.direction == 1 else "SELL"
        
        system_prompt = (
            "You are an expert Quantitative Trading Analyst and Risk Critic. "
            "Your job is to review technical trading signals and market context, "
            "and decide whether to confirm the trade, reject it, or pass. "
            "You MUST return valid JSON matching this schema: "
            '{"decision": "BUY"|"SELL"|"PASS", "confidence": float, "reasoning": str, "risk_assessment": str}'
        )
        
        prompt = (
            f"Technical Signal: {direction_str} (Score: {technical_signal.score})\n"
            f"Technical Reason: {technical_signal.reason}\n"
            f"Market Context:\n{market_context}\n"
            "Evaluate this trade."
        )
        
        try:
            response = self.provider.generate(prompt, system_prompt, json_mode=True)
            decision_data = json.loads(response)
            ai_decision = AIDecisionSchema(**decision_data)
            
            # Reconcile AI decision with technical signal
            if ai_decision.decision == "PASS":
                logger.info(f"AI rejected trade: {ai_decision.reasoning}")
                return Signal(direction=0, score=0.0, reason=ai_decision.reasoning, invalidation="")
                
            ai_direction = 1 if ai_decision.decision == "BUY" else -1
            
            if ai_direction != technical_signal.direction:
                logger.warning(f"AI disagrees with technical signal (Tech: {direction_str}, AI: {ai_decision.decision})")
                return Signal(direction=0, score=0.0, reason="AI disagreement", invalidation="")
                
            # If they agree, boost confidence
            final_score = (technical_signal.score + ai_decision.confidence) / 2.0
            
            return Signal(
                direction=technical_signal.direction,
                score=final_score,
                reason=f"{technical_signal.reason} | AI: {ai_decision.reasoning}",
                invalidation=technical_signal.invalidation
            )
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Fallback: trust the technical signal but reduce score slightly
            return Signal(
                direction=technical_signal.direction,
                score=technical_signal.score * 0.9,
                reason=f"{technical_signal.reason} (AI unavailable/invalid)",
                invalidation=technical_signal.invalidation
            )
        except Exception as e:
            logger.error(f"LLM Provider error: {e}")
            return Signal(
                direction=technical_signal.direction,
                score=technical_signal.score * 0.9,
                reason=f"{technical_signal.reason} (AI error)",
                invalidation=technical_signal.invalidation
            )
