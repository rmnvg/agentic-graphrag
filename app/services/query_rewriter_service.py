"""Service for improving a retrieval query after answer-judge feedback."""

import logging

from app.prompts.query_rewriter_prompt import build_query_rewriter_prompt
from app.services.llms import BaseLLM, LLMError, get_llm

logger = logging.getLogger(__name__)


class QueryRewriterService:
    """Rewrite queries for retrieval while safely preserving the original on failure."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        """Create a rewriter with an optional injected LLM for testing."""
        self._llm = llm

    def rewrite(
        self,
        original_question: str,
        previous_answer: str,
        judge_reason: str,
        retry_count: int,
    ) -> str:
        """Return a retrieval query, falling back to the original question safely."""
        try:
            logger.info("Query rewrite started (retry_count=%d).", retry_count)
            prompt = build_query_rewriter_prompt(
                original_question=original_question,
                previous_answer=previous_answer,
                judge_reason=judge_reason,
                retry_count=retry_count,
            )
            active_llm = self._llm or get_llm()
            rewritten_query = active_llm.generate(prompt).strip()
            if not rewritten_query:
                raise ValueError("Query rewriter returned an empty query.")

            logger.info("Query rewrite completed (retry_count=%d).", retry_count)
            return rewritten_query
        except (LLMError, ValueError, TypeError):
            logger.exception("Query rewrite failed; using the original question.")
            return original_question
        except Exception:
            logger.exception("Unexpected query rewrite failure; using the original question.")
            return original_question
