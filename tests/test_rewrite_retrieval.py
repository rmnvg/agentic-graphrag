"""Tests that rewritten queries affect retrieval without mutating the user question."""

from typing import Any

from app.graph.nodes import build_retrieve_node


def test_retrieve_node_uses_rewritten_query() -> None:
    captured_queries: list[str] = []

    def fake_retrieval(query: str, top_k: int) -> dict[str, Any]:
        captured_queries.append(query)
        assert top_k == 5
        return {"query": query, "matches": [{"chunk_id": "chunk-1", "text": "Context"}]}

    original_question = "Explain it."
    rewritten_query = "Explain Apache Kafka according to the uploaded document."
    result = build_retrieve_node(fake_retrieval)(
        {
            "question": original_question,
            "original_question": original_question,
            "rewritten_query": rewritten_query,
        }
    )

    assert captured_queries == [rewritten_query]
    assert result == {"retrieved_chunks": [{"chunk_id": "chunk-1", "text": "Context"}]}
