"""
core/ai/provider.py — LLM Provider abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        """Generate text from the LLM."""
        ...
        
class MockLLMProvider(LLMProvider):
    """A mock LLM provider for testing."""
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        
    def generate(self, prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        # Default mock response
        default_response = json.dumps({
            "decision": "BUY",
            "confidence": 85.0,
            "reasoning": "Strong bullish momentum.",
            "risk_assessment": "Acceptable risk."
        }) if json_mode else "Bullish momentum is strong."
        
        return self.responses.get(prompt, default_response)
