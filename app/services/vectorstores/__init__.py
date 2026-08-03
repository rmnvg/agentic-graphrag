"""Vector store adapters used by document indexing."""

from app.services.vectorstores.base_vectorstore import (
    BaseVectorStore,
    VectorPoint,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorStoreOperationError,
)
from app.services.vectorstores.qdrant_vectorstore import (
    QdrantVectorStore,
    get_qdrant_vectorstore,
)


def get_vectorstore() -> BaseVectorStore:
    """Return the configured default vector-store implementation."""
    return get_qdrant_vectorstore()

__all__ = [
    "BaseVectorStore",
    "QdrantVectorStore",
    "VectorPoint",
    "VectorStoreConfigurationError",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "VectorStoreOperationError",
    "get_qdrant_vectorstore",
    "get_vectorstore",
]
