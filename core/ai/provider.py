"""
core/ai/provider.py — LLM provider abstraction: Local endpoint, OpenAI-compatible
cloud, or nothing at all.

There is deliberately no built-in "embedded" reasoning engine. The one that used
to live here matched keywords in the prompt and returned hand-written verdicts
with hand-written confidence numbers (86.0, 91.0, 92.0). Those numbers were
printed and logged as AI confidence, but nothing had measured anything: they
were constants chosen to read well. That is the same failure mode the NN gate
and ``AIEnsemble.predict`` exist to prevent — a component that votes forever
without a measured edge.

The honest answer when no LLM is configured is *no answer*. ``get_llm_provider``
returns ``None``, and every caller already treats a missing or failing provider
as "no AI opinion" and falls back to its deterministic path (the structural
SL/TP, the technical confluence score). Absence is a valid, honest state; a
fabricated opinion is not.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import logging
import re
import requests

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No LLM is configured, or the configured one could not be reached.

    Callers catch this (they already catch ``Exception`` around every
    ``generate`` call) and fall back to their non-AI path. It is raised rather
    than answered so that no caller can mistake silence for a verdict.
    """


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        """Generate text from the LLM."""
        ...


class MockLLMProvider(LLMProvider):
    """A mock LLM provider for testing."""
    def __init__(self, responses: Optional[Dict[str, str]] = None):
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


class _HTTPChatProvider(LLMProvider):
    """Shared transport for the OpenAI-compatible ``/chat/completions`` shape.

    On failure it raises :class:`LLMUnavailable` unless an explicit fallback
    provider was supplied. There is no implicit fallback: a provider that
    cannot be reached must not be quietly replaced by something that invents an
    answer.
    """

    _DEFAULT_BASE_URL = "http://localhost:1234/v1"
    _DEFAULT_MODEL = "local-model"
    _DEFAULT_API_KEY = "local"

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
        timeout: float = 3.0,
        fallback_provider: Optional[LLMProvider] = None,
    ):
        self.base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self._DEFAULT_MODEL
        self.api_key = api_key or self._DEFAULT_API_KEY
        self.timeout = timeout
        self.fallback = fallback_provider

    def generate(self, prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if json_mode:
                return _clean_json_markdown(content)
            return content
        except Exception as exc:
            if self.fallback is not None:
                logger.warning("%s failed at %s (%s). Using the supplied fallback provider.",
                               type(self).__name__, endpoint, exc)
                return self.fallback.generate(prompt, system_prompt, json_mode=json_mode)
            logger.warning("%s failed at %s (%s). No AI opinion for this call.",
                           type(self).__name__, endpoint, exc)
            raise LLMUnavailable(f"{type(self).__name__} failed at {endpoint}: {exc}") from exc


class LocalEndpointProvider(_HTTPChatProvider):
    """
    Connects to any local OpenAI-compatible endpoint (LM Studio, vLLM, LocalAI, etc.).
    Example URL: http://localhost:1234/v1
    """

    _DEFAULT_BASE_URL = "http://localhost:1234/v1"
    _DEFAULT_MODEL = "local-model"
    _DEFAULT_API_KEY = "lm-studio"

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        api_key: str = "lm-studio",
        timeout: float = 3.0,
        fallback_provider: Optional[LLMProvider] = None,
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key,
                         timeout=timeout, fallback_provider=fallback_provider)


class OpenAICompatibleProvider(_HTTPChatProvider):
    """
    Connects to any OpenAI-compatible HTTP API:
    - Cloud: OpenAI, DeepSeek, Groq, Mistral, Together, OpenRouter
    - Local: LM Studio, vLLM, LocalAI
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _DEFAULT_MODEL = "gpt-4o-mini"
    _DEFAULT_API_KEY = "local"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        timeout: float = 4.0,
        fallback_provider: Optional[LLMProvider] = None,
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key,
                         timeout=timeout, fallback_provider=fallback_provider)


def _clean_json_markdown(raw: str) -> str:
    """Strip markdown code block wrappers (```json ... ```) from LLM responses."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


#: Names that mean "no LLM". ``embedded``/``builtin``/``internal`` are kept as
#: accepted aliases so a stale ``.env`` or a stored dashboard config keeps
#: loading — they now resolve to no provider instead of the deleted rule engine.
NO_PROVIDER_NAMES = ("", "none", "off", "disabled", "embedded", "builtin", "internal")


def get_llm_provider(
    provider_name: str = "none",
    base_url: str = "http://localhost:1234/v1",
    model: str = "",
    timeout: float = 3.0,
    api_key: str = "",
) -> Optional[LLMProvider]:
    """The configured LLM provider, or ``None`` when no LLM is configured.

    ``None`` is the default and the answer for any unrecognised name. Callers
    must treat it as "no AI opinion available" and use their deterministic
    path; there is no built-in engine to fall back to.
    """
    p = str(provider_name).strip().lower()
    if p in NO_PROVIDER_NAMES:
        return None
    if p in ("local_endpoint", "local", "lmstudio", "vllm"):
        return LocalEndpointProvider(base_url=base_url, model=model or "local-model",
                                     timeout=timeout, api_key=api_key)
    if p in ("openai", "openai_compatible", "deepseek", "groq", "cloud"):
        return OpenAICompatibleProvider(base_url=base_url or "https://api.openai.com/v1",
                                        model=model or "gpt-4o-mini", api_key=api_key,
                                        timeout=timeout)
    if p in ("mock", "test"):
        return MockLLMProvider()
    logger.warning("Unknown llm_provider %r; running with no LLM (no AI opinion).", provider_name)
    return None
