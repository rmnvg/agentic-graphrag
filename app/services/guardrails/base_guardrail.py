"""Provider-neutral interfaces for input and output safety classification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

GuardrailDecision = Literal["allow", "block", "skipped"]


class GuardrailError(Exception):
    """Base exception for guardrail provider failures."""


class GuardrailConfigurationError(GuardrailError):
    """Raised when safeguard provider settings are missing or invalid."""


class GuardrailOperationError(GuardrailError):
    """Raised when a safety classification cannot be completed."""


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Normalized decision returned by every guardrail implementation."""

    decision: GuardrailDecision
    reason: str | None = None
    category: str | None = None


class BaseGuardrail(ABC):
    """Interface for safety providers that inspect user input and model output."""

    @abstractmethod
    def evaluate_input(self, question: str) -> GuardrailResult:
        """Classify one user question before retrieval or generation."""

    @abstractmethod
    def evaluate_output(self, question: str, answer: str) -> GuardrailResult:
        """Classify one generated answer before it is returned to the user."""
