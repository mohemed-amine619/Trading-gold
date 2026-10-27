"""
tests/unit/test_llm_provider_embedded.py — Ollama is gone; the embedded engine is the default.

Guards:
1. The OllamaProvider class no longer exists in core/ai/provider.py.
2. Asking for "ollama" (a name a stale .env may still carry) resolves to the
   embedded provider rather than attempting an HTTP call to localhost:11434.
3. The settings enum no longer offers an OLLAMA choice.
"""

from core.ai import provider
from core.ai.provider import EmbeddedReasoningProvider, get_llm_provider
from core.settings import LLMProvider


def test_ollama_provider_class_is_removed():
    assert not hasattr(provider, "OllamaProvider")


def test_ollama_name_resolves_to_embedded():
    assert isinstance(get_llm_provider("ollama"), EmbeddedReasoningProvider)


def test_default_name_is_embedded():
    assert isinstance(get_llm_provider("embedded"), EmbeddedReasoningProvider)


def test_settings_enum_has_no_ollama_choice():
    assert "ollama" not in {member.value for member in LLMProvider}
