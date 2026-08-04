"""Groq implementation of the answer-generation interface."""

import os
from functools import lru_cache
from typing import Any

from groq import Groq

from app.services.llms.base_llm import (
    BaseLLM,
    LLMConfigurationError,
    LLMGenerationError,
)


class GroqLLM(BaseLLM):
    """Generate grounded RAG answers through Groq Chat Completions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Groq | None = None,
    ) -> None:
        """Create a Groq adapter from explicit or environment configuration.

        Args:
            api_key: Groq API key. Defaults to ``GROQ_API_KEY``.
            model: Groq model name. Defaults to ``GROQ_MODEL``.
            client: Optional client injection point for tests.
        """
        configured_api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        configured_model = model if model is not None else os.getenv("GROQ_MODEL")

        if not configured_api_key:
            raise LLMConfigurationError("GROQ_API_KEY must be configured.")
        if not configured_model:
            raise LLMConfigurationError("GROQ_MODEL must be configured.")

        self._model = configured_model
        self._client: Any = client or Groq(api_key=configured_api_key)

    def generate(self, prompt: str) -> str:
        """Generate a single non-streaming answer using the configured Groq model."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer questions only from supplied document context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = completion.choices[0].message.content
        except Exception as exc:
            raise LLMGenerationError("Groq request failed.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMGenerationError("Groq returned an empty answer.")

        return content.strip()


@lru_cache(maxsize=1)
def get_groq_llm() -> GroqLLM:
    """Return a cached Groq client adapter reused across requests."""
    return GroqLLM()
