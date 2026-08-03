"""Abstractions for vector database implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class VectorStoreError(Exception):
    """Base exception for vector store failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector store configuration is missing or invalid."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector database cannot be reached."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a vector database operation fails."""


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """A vector and its searchable payload."""

    point_id: str
    vector: list[float]
    payload: dict[str, Any]


class BaseVectorStore(ABC):
    """Interface implemented by vector database adapters."""

    @abstractmethod
    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create the collection when it does not already exist."""

    @abstractmethod
    def upsert_points(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Insert or update points by their stable identifiers."""
