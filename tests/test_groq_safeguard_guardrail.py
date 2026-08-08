"""Tests for strict Groq Safeguard response parsing."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.services.guardrails import GuardrailOperationError
from app.services.guardrails.groq_safeguard_guardrail import (
    GroqSafeguardGuardrail,
    _parse_classification,
)


class FakeCompletions:
    """Capture one Groq request and return a configured JSON response."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.request: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        """Return a minimal chat-completion-shaped result."""
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeGroqClient:
    """Expose the nested chat completion interface used by the adapter."""

    def __init__(self, content: str) -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_safeguard_parses_allowed_json() -> None:
    result = _parse_classification(
        '{"violation":false,"category":null,"rationale":"Benign request."}'
    )

    assert result.decision == "allow"
    assert result.category is None


def test_safeguard_parses_blocked_json_with_category() -> None:
    result = _parse_classification(
        '{"violation":true,"category":"Prompt Injection",'
        '"rationale":"Attempts to override system instructions."}'
    )

    assert result.decision == "block"
    assert result.category == "Prompt Injection"


def test_safeguard_accepts_documented_fenced_json() -> None:
    result = _parse_classification(
        '```json\n{"violation":1,"category":"Direct Override",'
        '"rationale":"Override attempt."}\n```'
    )

    assert result.decision == "block"


def test_safeguard_rejects_malformed_json() -> None:
    with pytest.raises(GuardrailOperationError):
        _parse_classification("not json")


def test_safeguard_sends_policy_and_untrusted_content_separately() -> None:
    client = FakeGroqClient(
        '{"violation":false,"category":null,"rationale":"Allowed."}'
    )
    guardrail = GroqSafeguardGuardrail(
        api_key="test-key",
        model="openai/gpt-oss-safeguard-20b",
        client=client,  # type: ignore[arg-type]
    )

    result = guardrail.evaluate_input("Explain this document")

    request = client.completions.request
    assert result.decision == "allow"
    assert request["model"] == "openai/gpt-oss-safeguard-20b"
    assert request["messages"][0]["role"] == "system"
    assert "GraphRAG Safety Policy" in request["messages"][0]["content"]
    assert request["messages"][1]["role"] == "user"
    assert "<user_input>" in request["messages"][1]["content"]
