"""Coordinate query embedding generation and vector-store retrieval."""

import logging
from typing import Any

from app.services.embeddings import BaseEmbedder, get_sentence_transformer_embedder
from app.services.retrieval import BaseRetriever, RetrieverError, get_retriever
from app.services.reranking_service import RerankingService, get_reranking_service

logger = logging.getLogger(__name__)


class SemanticRetrievalError(Exception):
    """Base exception for semantic retrieval pipeline failures."""


class InvalidSearchQueryError(SemanticRetrievalError):
    """Raised when a search query is empty after normalization."""


class QueryEmbeddingFailedError(SemanticRetrievalError):
    """Raised when the query cannot be embedded."""


class RetrievalFailedError(SemanticRetrievalError):
    """Raised when the vector database search cannot be completed."""


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    embedder: BaseEmbedder | None = None,
    retriever: BaseRetriever | None = None,
    reranking_service: RerankingService | None = None,
) -> dict[str, Any]:
    """Return the most relevant indexed chunks for a natural-language query.

    This function intentionally stops at semantic retrieval. It does not build a
    prompt, call an LLM, or generate an answer.

    Args:
        query: User's natural-language question.
        top_k: Final number of reranked chunks to return.
        embedder: Optional embedder injection for tests.
        retriever: Optional retrieval backend injection for tests.
        reranking_service: Optional reranking coordinator injection for tests.

    Returns:
        Query and ranked RAG chunks ready for a later answer-generation stage.

    Raises:
        InvalidSearchQueryError: If query is empty or top_k is invalid.
        QueryEmbeddingFailedError: If query embedding generation fails.
        RetrievalFailedError: If Qdrant retrieval fails.
    """
    normalized_query = _normalize_query(query)
    active_reranking_service = reranking_service or get_reranking_service()
    _validate_top_k(top_k, active_reranking_service.max_candidates)

    logger.info("Incoming semantic search query received (length=%d, top_k=%d).", len(normalized_query), top_k)
    logger.info("Query embedding generation started.")

    try:
        active_embedder = embedder or get_sentence_transformer_embedder()
        embeddings = active_embedder.embed_texts([normalized_query], batch_size=1)
        query_vector = _extract_query_vector(embeddings)
    except Exception as exc:
        logger.exception("Query embedding generation failed.")
        raise QueryEmbeddingFailedError("Unable to generate query embedding.") from exc

    logger.info("Query embedding generation completed.")
    logger.info("Hybrid retrieval started.")

    try:
        active_retriever = retriever or get_retriever()
        matches = active_retriever.retrieve(
            query=normalized_query,
            query_vector=query_vector,
            top_k=active_reranking_service.max_candidates,
        )
    except RetrieverError as exc:
        logger.exception("Qdrant search failed.")
        raise RetrievalFailedError("Unable to retrieve relevant document chunks.") from exc
    except Exception as exc:
        logger.exception("Semantic retrieval failed.")
        raise RetrievalFailedError("Unable to retrieve relevant document chunks.") from exc

    logger.info("Retrieved %d RRF chunks for reranking.", len(matches))
    reranked_matches = active_reranking_service.rerank(
        question=normalized_query,
        retrieved_chunks=matches,
        final_top_k=top_k,
    )
    logger.info("Semantic search completed.")

    return {
        "query": normalized_query,
        "matches": [
            {
                "score": match.score,
                "document_id": match.document_id,
                "chunk_id": match.chunk_id,
                "page": match.page,
                "pages": match.pages,
                "section": match.section,
                "text": match.text,
                "token_count": match.token_count,
                "metadata": match.metadata,
            }
            for match in reranked_matches
        ],
    }


def _normalize_query(query: str) -> str:
    """Trim a query and reject empty or non-string values."""
    if not isinstance(query, str):
        raise InvalidSearchQueryError("Query must be a non-empty string.")

    normalized_query = query.strip()
    if not normalized_query:
        raise InvalidSearchQueryError("Query must not be empty.")

    return normalized_query


def _validate_top_k(top_k: int, max_candidates: int) -> None:
    """Validate the requested result limit before invoking dependencies."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise InvalidSearchQueryError("top_k must be a positive integer.")
    if top_k > max_candidates:
        raise InvalidSearchQueryError(
            f"top_k must not exceed the rerank candidate limit ({max_candidates})."
        )


def _extract_query_vector(embeddings: list[list[float]]) -> list[float]:
    """Validate one query embedding returned by the shared embedder."""
    if len(embeddings) != 1 or not embeddings[0]:
        raise QueryEmbeddingFailedError("Embedder did not return a query vector.")

    return embeddings[0]
