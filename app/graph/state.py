"""Typed state shared by the minimal RAG LangGraph workflow."""

from typing import Any, Literal, NotRequired, TypedDict


class RAGGraphState(TypedDict):
    """State required to orchestrate one retrieval-and-generation request."""

    question: str
    original_question: NotRequired[str]
    rewritten_query: NotRequired[str]
    route: NotRequired[Literal["retrieve", "direct"]]
    retrieved_chunks: NotRequired[list[dict[str, Any]]]
    answer: NotRequired[str]
    sources: NotRequired[list[dict[str, Any]]]
    input_guard_decision: NotRequired[Literal["allow", "block", "skipped"]]
    output_guard_decision: NotRequired[Literal["allow", "block", "skipped"]]
    guardrail_reason: NotRequired[str | None]
    judge_result: NotRequired[Literal["accept", "retry"]]
    judge_reason: NotRequired[str]
    retry_count: NotRequired[int]
