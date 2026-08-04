"""Semantic retrieval interfaces and backend implementations."""

from app.services.retrieval.base_retriever import (
    BaseRetriever,
    RetrievedChunk,
    RetrieverConfigurationError,
    RetrieverConnectionError,
    RetrieverError,
    RetrieverOperationError,
)
from app.services.retrieval.qdrant_retriever import (
    QdrantRetriever,
    get_qdrant_retriever,
)


def get_retriever() -> BaseRetriever:
    """Return the configured default semantic retrieval implementation."""
    return get_qdrant_retriever()


__all__ = [
    "BaseRetriever",
    "QdrantRetriever",
    "RetrievedChunk",
    "RetrieverConfigurationError",
    "RetrieverConnectionError",
    "RetrieverError",
    "RetrieverOperationError",
    "get_qdrant_retriever",
    "get_retriever",
]
