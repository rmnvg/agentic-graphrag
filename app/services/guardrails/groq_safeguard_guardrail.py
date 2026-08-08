"""Groq Safeguard implementation for policy-based safety classification."""

import os
from typing import Any, Literal

from groq import Groq
from pydantic import BaseModel, ConfigDict, ValidationError

from app.prompts.groq_safeguard_prompt import (
    GuardrailStage,
    build_safeguard_content,
    build_safeguard_policy,
)
from app.services.guardrails.base_guardrail import (
    BaseGuardrail,
    GuardrailConfigurationError,
    GuardrailOperationError,
    GuardrailResult,
)

DEFAULT_GUARDRAIL_MODEL = "openai/gpt-oss-safeguard-20b"


class SafeguardClassification(BaseModel):
    """Strict JSON contract requested from Groq Safeguard."""

    model_config = ConfigDict(extra="forbid", strict=True)

    violation: bool | Literal[0, 1]
    category: str | None
    rationale: str


class GroqSafeguardGuardrail(BaseGuardrail):
    """Classify RAG inputs and outputs with Groq's policy-following safeguard."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 2,
        client: Groq | None = None,
    ) -> None:
        """Create a safeguard adapter from explicit or environment settings."""
        configured_api_key = api_key or os.getenv("GROQ_API_KEY")
        configured_model = model or os.getenv("GUARDRAIL_MODEL", DEFAULT_GUARDRAIL_MODEL)
        if not configured_api_key:
            raise GuardrailConfigurationError("GROQ_API_KEY must be configured.")
        if not configured_model:
            raise GuardrailConfigurationError("GUARDRAIL_MODEL must be configured.")

        self._model = configured_model
        self._client: Any = client or Groq(
            api_key=configured_api_key,
            timeout=timeout_seconds,
        )

    def evaluate_input(self, question: str) -> GuardrailResult:
        """Classify user input before it reaches retrieval or generation."""
        return self._evaluate(stage="input", question=question)

    def evaluate_output(self, question: str, answer: str) -> GuardrailResult:
        """Classify generated output before it reaches the API response."""
        return self._evaluate(stage="output", question=question, answer=answer)

    def _evaluate(
        self,
        stage: GuardrailStage,
        question: str,
        answer: str | None = None,
    ) -> GuardrailResult:
        """Request and normalize one strict safeguard classification."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": build_safeguard_policy(stage)},
                    {
                        "role": "user",
                        "content": build_safeguard_content(stage, question, answer),
                    },
                ],
                temperature=0,
                max_completion_tokens=256,
            )
            content = completion.choices[0].message.content
        except Exception as exc:
            raise GuardrailOperationError("Groq Safeguard request failed.") from exc

        if not isinstance(content, str) or not content.strip():
            raise GuardrailOperationError("Groq Safeguard returned an empty classification.")
        return _parse_classification(content)


def _parse_classification(content: str) -> GuardrailResult:
    """Parse strict safeguard JSON, accepting an optional JSON markdown fence."""
    try:
        classification = SafeguardClassification.model_validate_json(
            _remove_json_fence(content)
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise GuardrailOperationError(
            "Groq Safeguard returned an invalid classification."
        ) from exc

    if bool(classification.violation):
        return GuardrailResult(
            decision="block",
            reason=classification.rationale,
            category=classification.category,
        )
    return GuardrailResult(
        decision="allow",
        reason=classification.rationale,
        category=classification.category,
    )


def _remove_json_fence(content: str) -> str:
    """Remove one complete JSON code fence without altering JSON content."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3 or lines[0].lower() not in {"```", "```json"} or lines[-1] != "```":
        raise ValueError("Malformed JSON code fence.")
    return "\n".join(lines[1:-1]).strip()
