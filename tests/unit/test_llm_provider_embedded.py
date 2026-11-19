"""
tests/unit/test_llm_provider_embedded.py — there is no built-in reasoning engine.

Ollama was removed earlier; the "embedded" rule engine that replaced it has now
been removed too. It matched keywords in the prompt and answered with
hand-written confidence numbers (86.0, 91.0, 92.0) that were logged as AI
confidence without anything having been measured.

Guards:
1. Neither ``OllamaProvider`` nor ``EmbeddedReasoningProvider`` exists.
2. Every name that means "no LLM" — including the stale ``embedded``/``ollama``
   values a saved ``.env`` may still carry — resolves to ``None`` rather than to
   a fabricating stand-in or an HTTP call to localhost:11434.
3. Real provider names still resolve to real providers.
4. A provider that cannot be reached raises instead of silently falling back.
"""

import pytest

from core.ai import provider as provider_module
from core.ai.provider import (
    LLMUnavailable,
    LocalEndpointProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    get_llm_provider,
)
from core.settings import LLMProvider


def test_ollama_provider_class_is_removed():
    assert not hasattr(provider_module, "OllamaProvider")


def test_embedded_reasoning_provider_class_is_removed():
    """The fabricating rule engine must not come back."""
    assert not hasattr(provider_module, "EmbeddedReasoningProvider")


@pytest.mark.parametrize("name", ["", "none", "off", "disabled",
                                  "embedded", "builtin", "internal"])
def test_no_llm_names_resolve_to_none(name):
    assert get_llm_provider(name) is None


def test_stale_ollama_name_resolves_to_none():
    """A stale .env carrying 'ollama' must not attempt localhost:11434."""
    assert get_llm_provider("ollama") is None


def test_default_is_no_provider():
    assert get_llm_provider() is None


def test_real_provider_names_still_resolve():
    assert isinstance(get_llm_provider("local_endpoint"), LocalEndpointProvider)
    assert isinstance(get_llm_provider("openai_compatible"), OpenAICompatibleProvider)
    assert isinstance(get_llm_provider("mock"), MockLLMProvider)


def test_unreachable_provider_raises_instead_of_fabricating():
    """No implicit fallback: an unreachable endpoint must surface as
    LLMUnavailable so callers take their deterministic path."""
    p = LocalEndpointProvider(base_url="http://localhost:59999/v1", timeout=0.5)
    with pytest.raises(LLMUnavailable):
        p.generate("ping", "system", json_mode=True)


def test_explicit_fallback_is_still_honoured():
    """An explicitly supplied fallback is used; only the implicit one is gone."""
    p = LocalEndpointProvider(
        base_url="http://localhost:59999/v1",
        timeout=0.5,
        fallback_provider=MockLLMProvider({"ping": "fallback-answer"}),
    )
    assert p.generate("ping", "system", json_mode=True) == "fallback-answer"


def test_settings_enum_has_no_ollama_choice():
    assert "ollama" not in {member.value for member in LLMProvider}
