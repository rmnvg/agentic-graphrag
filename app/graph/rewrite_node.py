"""LangGraph node that delegates query rewriting to QueryRewriterService."""

import logging
from typing import Callable

from app.graph.state import RAGGraphState
from app.services.query_rewriter_service import QueryRewriterService

logger = logging.getLogger(__name__)

QueryRewriterFactory = Callable[[], QueryRewriterService]


def build_rewrite_node(
    rewriter_factory: QueryRewriterFactory = QueryRewriterService,
) -> Callable[[RAGGraphState], dict[str, str]]:
    """Build a thin node that rewrites the next retrieval query."""

    def rewrite_node(state: RAGGraphState) -> dict[str, str]:
        """Delegate rewriting while retaining the immutable original question."""
        original_question = state.get("original_question", state["question"])
        retry_count = state.get("retry_count", 0)
        rewritten_query = rewriter_factory().rewrite(
            original_question=original_question,
            previous_answer=state.get("answer", ""),
            judge_reason=state.get("judge_reason", ""),
            retry_count=retry_count,
        )
        logger.info(
            "LangGraph rewrite node completed (original_query=%r, rewritten_query=%r, "
            "retry_count=%d).",
            original_question,
            rewritten_query,
            retry_count,
        )
        return {"rewritten_query": rewritten_query}

    return rewrite_node
