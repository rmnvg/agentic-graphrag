"""Tests for strict judge parsing and safe fallback behavior."""

from app.services.judge_service import JudgeService
from app.services.llms.base_llm import BaseLLM, LLMGenerationError


class FakeLLM(BaseLLM):
    """Small fake LLM that returns a configured response or raises an error."""

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def generate(self, prompt: str) -> str:
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def test_judge_accepts_valid_accept_json() -> None:
    evaluation = JudgeService(
        FakeLLM('{"decision":"accept","reason":"Fully grounded."}')
    ).evaluate("Question", [], "Answer")

    assert evaluation.decision == "accept"
    assert evaluation.reason == "Fully grounded."


def test_judge_returns_retry_for_valid_retry_json() -> None:
    evaluation = JudgeService(
        FakeLLM('{"decision":"retry","reason":"Unsupported claim."}')
    ).evaluate("Question", [], "Answer")

    assert evaluation.decision == "retry"


def test_judge_defaults_to_accept_for_malformed_json() -> None:
    evaluation = JudgeService(FakeLLM("not json")).evaluate("Question", [], "Answer")

    assert evaluation.decision == "accept"
    assert "Judge unavailable" in evaluation.reason


def test_judge_defaults_to_accept_for_llm_failure() -> None:
    evaluation = JudgeService(FakeLLM(error=LLMGenerationError("failed"))).evaluate(
        "Question", [], "Answer"
    )

    assert evaluation.decision == "accept"
