"""LangGraph node that delegates answer evaluation to JudgeService."""

import logging
from typing import Any, Callable

from app.graph.state import RAGGraphState
from app.services.judge_service import JudgeService

logger = logging.getLogger(__name__)

JudgeFactory = Callable[[], JudgeService]


def build_judge_node(
    judge_factory: JudgeFactory = JudgeService,
) -> Callable[[RAGGraphState], dict[str, str]]:
    """Build a thin node that evaluates generated RAG answers."""

    def judge_node(state: RAGGraphState) -> dict[str, str]:
        """Delegate structured answer evaluation to the independent judge service."""
        evaluation = judge_factory().evaluate(
            question=state["question"],
            retrieved_chunks=state.get("retrieved_chunks", []),
            answer=state["answer"],
        )
        logger.info("LangGraph judge node selected '%s'.", evaluation.decision)
        return {"judge_result": evaluation.decision, "judge_reason": evaluation.reason}

    return judge_node
