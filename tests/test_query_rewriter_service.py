"""Tests for retrieval-oriented query rewriting and its safe fallback."""

from app.services.llms.base_llm import BaseLLM, LLMGenerationError
from app.services.query_rewriter_service import QueryRewriterService


class FakeLLM(BaseLLM):
    """Small fake provider for query-rewriter tests."""

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def generate(self, prompt: str) -> str:
        """Return the configured response or raise the configured failure."""
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def test_query_rewriter_returns_the_rewritten_query() -> None:
    rewritten_query = QueryRewriterService(
        FakeLLM("Explain Apache Kafka according to the uploaded document.")
    ).rewrite("Explain it.", "Kafka is a messaging platform.", "Too vague.", 1)

    assert rewritten_query == "Explain Apache Kafka according to the uploaded document."


def test_query_rewriter_falls_back_to_original_question_on_failure() -> None:
    original_question = "Explain it."

    rewritten_query = QueryRewriterService(
        FakeLLM(error=LLMGenerationError("Groq unavailable"))
    ).rewrite(original_question, "Previous answer", "Too vague.", 1)

    assert rewritten_query == original_question
