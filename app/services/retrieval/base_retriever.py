"""Abstractions for semantic retrieval implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class RetrieverError(Exception):
    """Base exception for retrieval backend failures."""


class RetrieverConfigurationError(RetrieverError):
    """Raised when retriever configuration is missing or invalid."""


class RetrieverConnectionError(RetrieverError):
    """Raised when the retrieval backend cannot be reached."""


class RetrieverOperationError(RetrieverError):
    """Raised when a retrieval operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One semantically similar chunk returned by a retrieval backend."""

    score: float
    document_id: str
    chunk_id: str
    page: int | None
    pages: list[int]
    section: str | None
    text: str
    token_count: int | None
    metadata: dict[str, Any]


class BaseRetriever(ABC):
    """Interface implemented by semantic retrieval backends."""

    @abstractmethod
    def retrieve(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the highest scoring chunks for a query vector."""
