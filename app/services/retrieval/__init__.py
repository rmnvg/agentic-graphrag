"""Semantic retrieval interfaces and backend implementations."""

from app.services.retrieval.base_retriever import (
    BaseRetriever,
    RetrievedChunk,
    RetrieverConfigurationError,
    RetrieverConnectionError,
    RetrieverError,
    RetrieverOperationError,
)


def get_retriever() -> BaseRetriever:
    """Return the configured default hybrid retrieval implementation."""
    from app.services.retrieval.hybrid_retriever import get_hybrid_retriever

    return get_hybrid_retriever()


def get_qdrant_retriever():
    """Return the Qdrant-only retriever for direct use or testing."""
    from app.services.retrieval.qdrant_retriever import get_qdrant_retriever as factory

    return factory()


__all__ = [
    "BaseRetriever",
    "RetrievedChunk",
    "RetrieverConfigurationError",
    "RetrieverConnectionError",
    "RetrieverError",
    "RetrieverOperationError",
    "get_qdrant_retriever",
    "get_retriever",
]
