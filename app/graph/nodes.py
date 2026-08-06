"""Thin LangGraph nodes that delegate to existing RAG services."""

import logging
from typing import Any, Callable

from app.graph.state import RAGGraphState
from app.graph.router import route_question
from app.services.direct_generation_service import generate_direct_answer
from app.services.rag_service import (
    generate_answer_from_chunks,
    retrieve_rag_chunks,
)
from app.services.retrieval_service import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

RetrievalFunction = Callable[[str, int], dict[str, Any]]
GenerationFunction = Callable[[str, list[dict[str, Any]]], dict[str, Any]]
DirectGenerationFunction = Callable[[str], dict[str, Any]]


def build_router_node() -> Callable[[RAGGraphState], dict[str, str]]:
    """Build a graph node that applies deterministic route selection."""

    def router_node(state: RAGGraphState) -> dict[str, str]:
        """Select the next node without retrieval or LLM work in the router."""
        route = route_question(state["question"])
        logger.info("LangGraph router selected '%s'.", route)
        return {"route": route}

    return router_node


def build_retrieve_node(
    retrieval_function: RetrievalFunction = retrieve_relevant_chunks,
) -> Callable[[RAGGraphState], dict[str, Any]]:
    """Build a graph node that delegates retrieval to the existing RAG service."""

    def retrieve_node(state: RAGGraphState) -> dict[str, Any]:
        """Retrieve supporting chunks without embedding or search logic in the node."""
        original_question = state.get("original_question", state["question"])
        retrieval_query = state.get("rewritten_query") or original_question
        logger.info(
            "LangGraph retrieve node started (original_query=%r, retrieval_query=%r).",
            original_question,
            retrieval_query,
        )
        retrieval_result = retrieve_rag_chunks(
            question=retrieval_query,
            retrieval_function=retrieval_function,
        )
        chunks = retrieval_result["matches"]
        logger.info("LangGraph retrieve node completed with %d chunks.", len(chunks))
        return {"retrieved_chunks": chunks}

    return retrieve_node


def build_generate_node(
    generation_function: GenerationFunction = generate_answer_from_chunks,
) -> Callable[[RAGGraphState], dict[str, Any]]:
    """Build a graph node that delegates prompting and Groq generation to RAG services."""

    def generate_node(state: RAGGraphState) -> dict[str, Any]:
        """Generate an answer without prompt or LLM communication logic in the node."""
        logger.info("LangGraph generate node started.")
        generation_result = generation_function(
            question=state["question"],
            matches=state.get("retrieved_chunks", []),
        )
        logger.info("LangGraph generate node completed.")
        return {
            "answer": generation_result["answer"],
            "sources": generation_result["sources"],
        }

    return generate_node


def build_direct_node(
    generation_function: DirectGenerationFunction = generate_direct_answer,
) -> Callable[[RAGGraphState], dict[str, Any]]:
    """Build a graph node for conversational requests without document retrieval."""

    def direct_node(state: RAGGraphState) -> dict[str, Any]:
        """Delegate direct LLM generation to the dedicated direct service."""
        logger.info("LangGraph direct generation node started.")
        generation_result = generation_function(state["question"])
        logger.info("LangGraph direct generation node completed.")
        return {
            "question": generation_result["question"],
            "answer": generation_result["answer"],
            "sources": generation_result["sources"],
        }

    return direct_node


def build_retry_node() -> Callable[[RAGGraphState], dict[str, int]]:
    """Build a node that records a bounded query-retrieval retry attempt."""

    def retry_node(state: RAGGraphState) -> dict[str, int]:
        """Increment retry state before query rewriting and fresh retrieval."""
        retry_count = state.get("retry_count", 0) + 1
        logger.info("LangGraph retry node recorded retry count %d.", retry_count)
        return {"retry_count": retry_count}

    return retry_node
