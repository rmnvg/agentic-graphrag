"""Tests that safety blocks terminate the LangGraph workflow at the right stage."""

from typing import Any

from app.graph.graph import build_rag_graph


def _unused_node(name: str):
    """Build a node that fails the test if a blocked branch reaches it."""

    def node(state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"{name} should not execute")

    return node


def test_input_block_ends_before_router() -> None:
    graph = build_rag_graph(
        input_guard_node=lambda state: {
            "input_guard_decision": "block",
            "answer": "Blocked",
            "sources": [],
        },
        router_node=_unused_node("router"),
    )

    result = graph.invoke({"question": "Unsafe question", "retry_count": 0})

    assert result["answer"] == "Blocked"
    assert result["sources"] == []


def test_output_block_ends_before_judge() -> None:
    graph = build_rag_graph(
        input_guard_node=lambda state: {"input_guard_decision": "allow"},
        router_node=lambda state: {"route": "retrieve"},
        retrieve_node=lambda state: {"retrieved_chunks": []},
        generate_node=lambda state: {"answer": "Unsafe answer", "sources": []},
        output_guard_node=lambda state: {
            "output_guard_decision": "block",
            "answer": "Blocked output",
            "sources": [],
        },
        judge_node=_unused_node("judge"),
    )

    result = graph.invoke({"question": "Question", "retry_count": 0})

    assert result["answer"] == "Blocked output"
    assert result["sources"] == []
