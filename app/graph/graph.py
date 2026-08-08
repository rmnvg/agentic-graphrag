"""Minimal LangGraph orchestration for the existing RAG pipeline."""

from functools import lru_cache
import logging
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_direct_node,
    build_generate_node,
    build_retrieve_node,
    build_retry_node,
    build_router_node,
)
from app.graph.input_guard_node import build_input_guard_node
from app.graph.judge_node import build_judge_node
from app.graph.output_guard_node import build_output_guard_node
from app.graph.router import GraphRoute
from app.graph.rewrite_node import build_rewrite_node
from app.graph.state import RAGGraphState

GraphNode = Callable[[RAGGraphState], dict[str, Any]]
MAX_GENERATION_RETRIES = 2
logger = logging.getLogger(__name__)


def build_rag_graph(
    router_node: GraphNode | None = None,
    retrieve_node: GraphNode | None = None,
    generate_node: GraphNode | None = None,
    direct_node: GraphNode | None = None,
    judge_node: GraphNode | None = None,
    retry_node: GraphNode | None = None,
    rewrite_node: GraphNode | None = None,
    input_guard_node: GraphNode | None = None,
    output_guard_node: GraphNode | None = None,
) -> Any:
    """Compile guarded RAG-with-judge and direct-generation workflow.

    Optional node injection keeps orchestration tests independent from external
    retrieval and LLM services.
    """
    workflow = StateGraph(RAGGraphState)
    workflow.add_node("input_guard", input_guard_node or build_input_guard_node())
    workflow.add_node("router", router_node or build_router_node())
    workflow.add_node("retrieve", retrieve_node or build_retrieve_node())
    workflow.add_node("generate", generate_node or build_generate_node())
    workflow.add_node("direct", direct_node or build_direct_node())
    workflow.add_node("judge", judge_node or build_judge_node())
    workflow.add_node("retry", retry_node or build_retry_node())
    workflow.add_node("rewrite", rewrite_node or build_rewrite_node())
    workflow.add_node("output_guard", output_guard_node or build_output_guard_node())
    workflow.add_edge(START, "input_guard")
    workflow.add_conditional_edges("input_guard", _route_after_input_guard, {
        "router": "router",
        "end": END,
    })
    workflow.add_conditional_edges("router", _route_from_state, {
        "retrieve": "retrieve",
        "direct": "direct",
    })
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "output_guard")
    workflow.add_conditional_edges("output_guard", _route_after_output_guard, {
        "judge": "judge",
        "end": END,
    })
    workflow.add_conditional_edges("judge", _route_after_judge, {
        "accept": END,
        "retry": "retry",
    })
    workflow.add_conditional_edges("retry", _route_after_retry, {
        "rewrite": "rewrite",
        "end": END,
    })
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("direct", "output_guard")
    return workflow.compile()


@lru_cache(maxsize=1)
def get_rag_graph() -> Any:
    """Return the reusable compiled production RAG workflow."""
    return build_rag_graph()


def invoke_rag_graph(question: str) -> dict[str, Any]:
    """Run one stateless RAG graph invocation and return its final state."""
    result = dict(
        get_rag_graph().invoke(
            {
                "question": question,
                "original_question": question,
                "retry_count": 0,
            }
        )
    )
    logger.info(
        "LangGraph workflow completed (route=%s, judge=%s, retries=%d).",
        result.get("route"),
        result.get("judge_result"),
        result.get("retry_count", 0),
    )
    return result


def _route_from_state(state: RAGGraphState) -> GraphRoute:
    """Read the route chosen by the router node for a conditional graph edge."""
    return state["route"]


def _route_after_input_guard(state: RAGGraphState) -> Literal["router", "end"]:
    """End blocked requests before router, retrieval, or generation can run."""
    return "end" if state["input_guard_decision"] == "block" else "router"


def _route_after_output_guard(state: RAGGraphState) -> Literal["judge", "end"]:
    """Judge safe RAG output, while safe direct output can end immediately."""
    if state["output_guard_decision"] == "block":
        return "end"
    return "judge" if state["route"] == "retrieve" else "end"


def _route_after_judge(state: RAGGraphState) -> Literal["accept", "retry"]:
    """Route accepted answers to END and rejected answers through retry handling."""
    return state["judge_result"]


def _route_after_retry(state: RAGGraphState) -> Literal["rewrite", "end"]:
    """Rewrite and retrieve again unless the retry limit has been reached."""
    if state["retry_count"] >= MAX_GENERATION_RETRIES:
        logger.info("LangGraph retry limit reached.")
        return "end"

    return "rewrite"
