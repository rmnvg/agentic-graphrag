"""LLM interfaces and provider implementations."""

from app.services.llms.base_llm import (
    BaseLLM,
    LLMConfigurationError,
    LLMError,
    LLMGenerationError,
)
from app.services.llms.groq_llm import GroqLLM, get_groq_llm


def get_llm() -> BaseLLM:
    """Return the configured default answer-generation provider."""
    return get_groq_llm()


__all__ = [
    "BaseLLM",
    "GroqLLM",
    "LLMConfigurationError",
    "LLMError",
    "LLMGenerationError",
    "get_groq_llm",
    "get_llm",
]
