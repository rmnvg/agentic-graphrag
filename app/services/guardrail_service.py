"""Feature-flagged orchestration for input and output safety checks."""

import logging
from functools import lru_cache
from time import perf_counter
from typing import Callable, Literal

from app.core.guardrail_settings import GuardrailFailMode, GuardrailSettings, get_guardrail_settings
from app.services.guardrails import (
    BaseGuardrail,
    GroqSafeguardGuardrail,
    GuardrailError,
    GuardrailResult,
)

BLOCKED_INPUT_RESPONSE = "I can't help with that request."
BLOCKED_OUTPUT_RESPONSE = "I can't provide that response."

logger = logging.getLogger(__name__)
GuardrailFactory = Callable[[float], BaseGuardrail]
GuardrailStage = Literal["input", "output"]


def _create_groq_safeguard(timeout_seconds: float) -> BaseGuardrail:
    """Create the default provider while passing its timeout explicitly."""
    return GroqSafeguardGuardrail(timeout_seconds=timeout_seconds)


class GuardrailService:
    """Apply optional safeguard checks without exposing provider details to callers."""

    def __init__(
        self,
        settings: GuardrailSettings,
        guardrail_factory: GuardrailFactory = _create_groq_safeguard,
    ) -> None:
        """Create a feature-flagged safety service with optional test injection."""
        self._settings = settings
        self._guardrail_factory = guardrail_factory
        self._guardrail: BaseGuardrail | None = None

    def check_input(self, question: str) -> GuardrailResult:
        """Evaluate input only when the global and input flags are enabled."""
        if not self._settings.enabled or not self._settings.input_enabled:
            return _skipped_result("input")
        return self._evaluate(stage="input", question=question)

    def check_output(self, question: str, answer: str) -> GuardrailResult:
        """Evaluate output only when the global and output flags are enabled."""
        if not self._settings.enabled or not self._settings.output_enabled:
            return _skipped_result("output")
        return self._evaluate(stage="output", question=question, answer=answer)

    def _evaluate(
        self,
        stage: GuardrailStage,
        question: str,
        answer: str | None = None,
    ) -> GuardrailResult:
        """Delegate classification and apply the configured provider-failure policy."""
        started_at = perf_counter()
        try:
            guardrail = self._guardrail or self._guardrail_factory(
                self._settings.timeout_seconds
            )
            self._guardrail = guardrail
            result = (
                guardrail.evaluate_input(question)
                if stage == "input"
                else guardrail.evaluate_output(question, answer or "")
            )
            logger.info(
                "Guardrail %s check completed (decision=%s, category=%s, latency=%.3f seconds).",
                stage,
                result.decision,
                result.category,
                perf_counter() - started_at,
            )
            return result
        except GuardrailError:
            logger.exception("Guardrail %s check failed.", stage)
        except Exception:
            logger.exception("Unexpected guardrail %s failure.", stage)

        return _failure_result(stage, self._fail_mode_for(stage))

    def _fail_mode_for(self, stage: GuardrailStage) -> GuardrailFailMode:
        """Return the configured failure policy for one guardrail stage."""
        return self._settings.input_fail_mode if stage == "input" else self._settings.output_fail_mode


def _skipped_result(stage: GuardrailStage) -> GuardrailResult:
    """Return a traceable no-op result for a disabled safety stage."""
    logger.info("Guardrail %s check skipped because it is disabled.", stage)
    return GuardrailResult(decision="skipped")


def _failure_result(stage: GuardrailStage, fail_mode: GuardrailFailMode) -> GuardrailResult:
    """Turn a provider failure into the configured availability or safety decision."""
    if fail_mode == "closed":
        return GuardrailResult(
            decision="block",
            reason="The safety check is temporarily unavailable.",
        )

    logger.warning("Guardrail %s failure allowed by fail-open policy.", stage)
    return GuardrailResult(
        decision="allow",
        reason="The safety check was unavailable; fail-open policy applied.",
    )


@lru_cache(maxsize=1)
def get_guardrail_service() -> GuardrailService:
    """Return the shared feature-flagged guardrail service for application requests."""
    return GuardrailService(settings=get_guardrail_settings())
