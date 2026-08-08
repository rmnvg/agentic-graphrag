"""Tests for feature-flagged safety checks and provider failure policies."""

from app.core.guardrail_settings import GuardrailSettings
from app.services.guardrail_service import GuardrailService
from app.services.guardrails import BaseGuardrail, GuardrailOperationError, GuardrailResult


class FakeGuardrail(BaseGuardrail):
    """Configurable safety provider fake with call tracking."""

    def __init__(
        self,
        input_result: GuardrailResult | None = None,
        output_result: GuardrailResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.input_result = input_result or GuardrailResult(decision="allow")
        self.output_result = output_result or GuardrailResult(decision="allow")
        self.error = error
        self.input_calls = 0
        self.output_calls = 0

    def evaluate_input(self, question: str) -> GuardrailResult:
        """Return a configured input result or provider failure."""
        self.input_calls += 1
        if self.error:
            raise self.error
        return self.input_result

    def evaluate_output(self, question: str, answer: str) -> GuardrailResult:
        """Return a configured output result or provider failure."""
        self.output_calls += 1
        if self.error:
            raise self.error
        return self.output_result


def _settings(
    *,
    enabled: bool = True,
    input_enabled: bool = True,
    output_enabled: bool = True,
    input_fail_mode: str = "open",
    output_fail_mode: str = "open",
) -> GuardrailSettings:
    """Build deterministic settings without reading process environment."""
    return GuardrailSettings(
        enabled=enabled,
        input_enabled=input_enabled,
        output_enabled=output_enabled,
        input_fail_mode=input_fail_mode,  # type: ignore[arg-type]
        output_fail_mode=output_fail_mode,  # type: ignore[arg-type]
        timeout_seconds=2,
    )


def test_disabled_input_guard_skips_provider_invocation() -> None:
    provider = FakeGuardrail()
    service = GuardrailService(
        _settings(input_enabled=False),
        guardrail_factory=lambda timeout: provider,
    )

    result = service.check_input("Hello")

    assert result.decision == "skipped"
    assert provider.input_calls == 0


def test_enabled_input_guard_returns_provider_block_decision() -> None:
    provider = FakeGuardrail(input_result=GuardrailResult(decision="block", category="S1"))
    service = GuardrailService(_settings(), guardrail_factory=lambda timeout: provider)

    result = service.check_input("Unsafe request")

    assert result.decision == "block"
    assert result.category == "S1"
    assert provider.input_calls == 1


def test_output_failure_uses_fail_open_policy() -> None:
    provider = FakeGuardrail(error=GuardrailOperationError("Provider unavailable"))
    service = GuardrailService(
        _settings(output_fail_mode="open"),
        guardrail_factory=lambda timeout: provider,
    )

    result = service.check_output("Question", "Answer")

    assert result.decision == "allow"
    assert "fail-open" in (result.reason or "")


def test_output_failure_uses_fail_closed_policy() -> None:
    provider = FakeGuardrail(error=GuardrailOperationError("Provider unavailable"))
    service = GuardrailService(
        _settings(output_fail_mode="closed"),
        guardrail_factory=lambda timeout: provider,
    )

    result = service.check_output("Question", "Answer")

    assert result.decision == "block"
