"""LangGraph node that checks generated answers before workflow completion."""

import logging
from typing import Callable

from app.graph.state import RAGGraphState
from app.services.guardrail_service import BLOCKED_OUTPUT_RESPONSE, GuardrailService, get_guardrail_service

logger = logging.getLogger(__name__)
GuardrailServiceFactory = Callable[[], GuardrailService]


def build_output_guard_node(
    guardrail_service_factory: GuardrailServiceFactory = get_guardrail_service,
) -> Callable[[RAGGraphState], dict[str, object]]:
    """Build a thin node that blocks unsafe generated output."""

    def output_guard_node(state: RAGGraphState) -> dict[str, object]:
        """Delegate output classification to the guardrail service."""
        result = guardrail_service_factory().check_output(
            question=state["question"],
            answer=state["answer"],
        )
        logger.info("LangGraph output guard decision=%s.", result.decision)
        if result.decision == "block":
            return {
                "output_guard_decision": result.decision,
                "guardrail_reason": result.reason,
                "answer": BLOCKED_OUTPUT_RESPONSE,
                "sources": [],
            }
        return {
            "output_guard_decision": result.decision,
            "guardrail_reason": result.reason,
        }

    return output_guard_node
