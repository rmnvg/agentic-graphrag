"""Provider-neutral contracts for chunk reranking implementations."""

from abc import ABC, abstractmethod

from app.services.retrieval.base_retriever import RetrievedChunk


class RerankerError(Exception):
    """Base exception raised by reranking implementations."""


class RerankerConfigurationError(RerankerError):
    """Raised when a reranker is configured with invalid settings."""


class RerankerOperationError(RerankerError):
    """Raised when a reranker cannot score candidate chunks."""


class BaseReranker(ABC):
    """Interface for components that reorder already retrieved chunks."""

    @abstractmethod
    def rerank(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return the supplied chunks reordered by relevance to ``question``."""
