"""Abstractions for exact-keyword retrieval implementations."""

from abc import ABC, abstractmethod

from app.services.retrieval.base_retriever import RetrievedChunk


class LexicalRetrieverError(Exception):
    """Base exception for lexical retrieval failures."""


class BaseLexicalRetriever(ABC):
    """Interface implemented by lexical retrievers such as BM25."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the highest-ranked chunks for an exact-keyword query."""
