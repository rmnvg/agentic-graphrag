"""Provider-neutral interface for answer-generation models."""

from abc import ABC, abstractmethod


class LLMError(Exception):
    """Base exception for LLM provider failures."""


class LLMConfigurationError(LLMError):
    """Raised when required LLM provider configuration is missing."""


class LLMGenerationError(LLMError):
    """Raised when an LLM provider cannot generate a usable response."""


class BaseLLM(ABC):
    """Interface implemented by LLM providers used for RAG generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate one non-streaming answer from a fully constructed prompt."""
