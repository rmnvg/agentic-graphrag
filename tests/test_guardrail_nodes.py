"""Tests for graph-node behavior around blocked guardrail results."""

from app.core.guardrail_settings import GuardrailSettings
from app.graph.input_guard_node import build_input_guard_node
from app.graph.output_guard_node import build_output_guard_node
from app.services.guardrail_service import (
    BLOCKED_INPUT_RESPONSE,
    BLOCKED_OUTPUT_RESPONSE,
    GuardrailService,
)
from app.services.guardrails import BaseGuardrail, GuardrailResult


class BlockingGuardrail(BaseGuardrail):
    """Safety fake that blocks both stages."""

    def evaluate_input(self, question: str) -> GuardrailResult:
        """Return a blocked input result."""
        return GuardrailResult(decision="block", reason="Policy")

    def evaluate_output(self, question: str, answer: str) -> GuardrailResult:
        """Return a blocked output result."""
        return GuardrailResult(decision="block", reason="Policy")


def _blocking_service() -> GuardrailService:
    """Build a service that consistently blocks, without external calls."""
    settings = GuardrailSettings(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_fail_mode="open",
        output_fail_mode="open",
        timeout_seconds=2,
    )
    return GuardrailService(settings, guardrail_factory=lambda timeout: BlockingGuardrail())


def test_input_guard_block_returns_safe_chat_response_fields() -> None:
    result = build_input_guard_node(_blocking_service)({"question": "Unsafe input"})

    assert result["input_guard_decision"] == "block"
    assert result["answer"] == BLOCKED_INPUT_RESPONSE
    assert result["sources"] == []


def test_output_guard_block_replaces_generated_answer() -> None:
    result = build_output_guard_node(_blocking_service)(
        {"question": "Question", "answer": "Unsafe answer"}
    )

    assert result["output_guard_decision"] == "block"
    assert result["answer"] == BLOCKED_OUTPUT_RESPONSE
    assert result["sources"] == []
