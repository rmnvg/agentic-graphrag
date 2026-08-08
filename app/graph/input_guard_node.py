"""LangGraph node that checks a question before routing it through the workflow."""

import logging
from typing import Callable

from app.graph.state import RAGGraphState
from app.services.guardrail_service import BLOCKED_INPUT_RESPONSE, GuardrailService, get_guardrail_service

logger = logging.getLogger(__name__)
GuardrailServiceFactory = Callable[[], GuardrailService]


def build_input_guard_node(
    guardrail_service_factory: GuardrailServiceFactory = get_guardrail_service,
) -> Callable[[RAGGraphState], dict[str, object]]:
    """Build a thin node that blocks unsafe questions before any RAG work begins."""

    def input_guard_node(state: RAGGraphState) -> dict[str, object]:
        """Delegate input classification to the guardrail service."""
        result = guardrail_service_factory().check_input(state["question"])
        logger.info("LangGraph input guard decision=%s.", result.decision)
        if result.decision == "block":
            return {
                "input_guard_decision": result.decision,
                "guardrail_reason": result.reason,
                "answer": BLOCKED_INPUT_RESPONSE,
                "sources": [],
            }
        return {
            "input_guard_decision": result.decision,
            "guardrail_reason": result.reason,
        }

    return input_guard_node
