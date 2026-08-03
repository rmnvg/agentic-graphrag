"""Qdrant implementation of the vector store interface."""

import logging
import os
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from app.services.vectorstores.base_vectorstore import (
    BaseVectorStore,
    VectorPoint,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreOperationError,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Persist dense vectors to a Qdrant collection using cosine distance."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        """Create a Qdrant adapter from explicit or environment configuration.

        Args:
            url: Qdrant server URL. Defaults to ``QDRANT_URL``.
            api_key: Optional Qdrant API key. Defaults to ``QDRANT_API_KEY``.
            client: Optional client injection point for tests.
        """
        configured_url = url or os.getenv("QDRANT_URL")
        configured_api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")

        if not configured_url:
            raise VectorStoreConfigurationError("QDRANT_URL must be configured.")

        self._client = client or QdrantClient(
            url=configured_url,
            api_key=configured_api_key or None,
        )

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a cosine-distance collection if it does not exist."""
        try:
            if self._client.collection_exists(collection_name=collection_name):
                return

            logger.info(
                "Creating Qdrant collection '%s' with vector size %d.",
                collection_name,
                vector_size,
            )
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise _translate_qdrant_error("ensure collection", exc) from exc

    def upsert_points(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Upsert points so retries update existing chunk IDs instead of duplicating."""
        if not points:
            return

        try:
            self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point.point_id,
                        vector=point.vector,
                        payload=point.payload,
                    )
                    for point in points
                ],
                wait=True,
            )
        except Exception as exc:
            raise _translate_qdrant_error("upsert points", exc) from exc


def _translate_qdrant_error(operation: str, error: Exception) -> VectorStoreError:
    """Translate Qdrant client exceptions without exposing internal details."""
    error_name = type(error).__name__.lower()

    if any(marker in error_name for marker in ("connection", "timeout", "responsehandling")):
        return VectorStoreConnectionError(f"Qdrant connection failed while attempting to {operation}.")

    return VectorStoreOperationError(f"Qdrant failed to {operation}.")


@lru_cache(maxsize=1)
def get_qdrant_vectorstore() -> QdrantVectorStore:
    """Return a reusable Qdrant adapter configured from environment variables."""
    return QdrantVectorStore()
