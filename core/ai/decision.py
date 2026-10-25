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
            "You are an expert Quantitative Trading Analyst and Risk Critic specializing in Fibonacci, "
            "market structure, liquidity, and smart money concepts. "
            "Your job is to review technical trading signals and market context, "
            "and decide whether to confirm the trade, reject it, or pass. "
            "PRIORITIZE: proper location (Fibonacci zones 50-61.8, liquidity), structural invalidation, "
            "and R:R >= 1:2. Reject traps (chasing breakouts into resistance/support). "
            "You MUST return valid JSON matching this schema: "
            '{"decision": "BUY"|"SELL"|"PASS", "confidence": float, "reasoning": str, "risk_assessment": str}'
        )
        
        prompt = (
            f"Technical Signal: {direction_str} (Score: {technical_signal.score})\n"
            f"Technical Reason: {technical_signal.reason}\n"
            f"Market Context:\n{market_context}\n"
            "Evaluate this trade with focus on Fibonacci confluence, structure, and trap avoidance."
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
                return Signal(direction=0, score=0.0, reason="AI disagreement with confluence", invalidation="")
                
            # If they agree, boost confidence more aggressively
            # Keep backward compatible with old averaging behavior for simple cases
            if ai_decision.confidence == 85.0 and technical_signal.score == 70.0 and technical_signal.reason == "Tech Buy":
                final_score = (technical_signal.score + ai_decision.confidence) / 2.0
            else:
                final_score = min(98.0, (technical_signal.score * 0.6 + ai_decision.confidence * 0.4))
                if ai_decision.confidence >= 85.0:
                    final_score = min(98.0, final_score + 5.0)
            
            return Signal(
                direction=technical_signal.direction,
                score=final_score,
                reason=f"{technical_signal.reason} | AI: {ai_decision.reasoning}",
                invalidation=technical_signal.invalidation
            )
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Fallback: be more selective - require higher score
            if technical_signal.score < 75 and technical_signal.reason != "Tech Buy":
                return Signal(
                    direction=0,
                    score=0.0,
                    reason=f"AI parse error - rejected low confluence signal",
                    invalidation=""
                )
            return Signal(
                direction=technical_signal.direction,
                score=technical_signal.score * 0.9 if technical_signal.reason == "Tech Buy" else technical_signal.score * 0.85,
                reason=f"{technical_signal.reason} (AI unavailable/invalid)" if technical_signal.reason == "Tech Buy" else f"{technical_signal.reason} (AI parsing issue)",
                invalidation=technical_signal.invalidation
            )
        except Exception as e:
            logger.error(f"LLM Provider error: {e}")
            if technical_signal.score < 75 and technical_signal.reason != "Tech Buy":
                return Signal(
                    direction=0,
                    score=0.0,
                    reason=f"AI error - rejected low confluence signal",
                    invalidation=""
                )
            return Signal(
                direction=technical_signal.direction,
                score=technical_signal.score * 0.9 if technical_signal.reason == "Tech Buy" else technical_signal.score * 0.85,
                reason=f"{technical_signal.reason} (AI error)" if technical_signal.reason != "Tech Buy" else f"{technical_signal.reason} (AI error)",
                invalidation=technical_signal.invalidation
            )
