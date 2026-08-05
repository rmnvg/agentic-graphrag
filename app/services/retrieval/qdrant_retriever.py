"""Qdrant-backed semantic chunk retrieval."""

import os
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient

from app.services.retrieval.base_retriever import (
    BaseRetriever,
    RetrievedChunk,
    RetrieverConfigurationError,
    RetrieverConnectionError,
    RetrieverError,
    RetrieverOperationError,
)

DOCUMENTS_COLLECTION = "documents"


class QdrantRetriever(BaseRetriever):
    """Retrieve nearest document chunks from Qdrant using cosine similarity."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        """Create a Qdrant retriever from explicit or environment configuration."""
        configured_url = url or os.getenv("QDRANT_URL")
        configured_api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")

        if not configured_url:
            raise RetrieverConfigurationError("QDRANT_URL must be configured.")

        self._client = client or QdrantClient(
            url=configured_url,
            api_key=configured_api_key or None,
        )

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Query Qdrant and map matching point payloads into stable result objects."""
        try:
            response = self._client.query_points(
                collection_name=DOCUMENTS_COLLECTION,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return [_to_retrieved_chunk(point) for point in response.points]
        except RetrieverError:
            raise
        except Exception as exc:
            raise _translate_qdrant_error(exc) from exc


def _to_retrieved_chunk(point: Any) -> RetrievedChunk:
    """Map one Qdrant scored point into the retrieval contract."""
    payload = point.payload or {}
    if not isinstance(payload, dict):
        raise RetrieverOperationError("Qdrant returned an invalid point payload.")

    document_id = payload.get("document_id")
    chunk_id = payload.get("chunk_id") or str(point.id)
    text = payload.get("text")

    if not all(isinstance(value, str) and value for value in (document_id, chunk_id, text)):
        raise RetrieverOperationError("Qdrant point is missing required chunk payload.")

    return RetrievedChunk(
        score=float(point.score),
        document_id=document_id,
        chunk_id=chunk_id,
        page=_optional_int(payload.get("page")),
        pages=_page_numbers(payload.get("pages")),
        section=_optional_string(payload.get("section")),
        text=text,
        token_count=_optional_int(payload.get("token_count")),
        metadata=_metadata(payload.get("metadata")),
    )


def _optional_int(value: Any) -> int | None:
    """Return an integer payload field when present and valid."""
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_numbers(value: Any) -> list[int]:
    """Normalize page payload values without failing a whole search result."""
    if not isinstance(value, list):
        return []

    return [page for item in value if (page := _optional_int(item)) is not None]


def _optional_string(value: Any) -> str | None:
    """Return a non-empty string payload field when present."""
    if not isinstance(value, str):
        return None

    return value or None


def _metadata(value: Any) -> dict[str, Any]:
    """Return metadata only when Qdrant stored it as an object."""
    return value if isinstance(value, dict) else {}


def _translate_qdrant_error(error: Exception) -> RetrieverError:
    """Translate Qdrant exceptions into backend-neutral retrieval errors."""
    error_name = type(error).__name__.lower()
    if any(marker in error_name for marker in ("connection", "timeout", "responsehandling")):
        return RetrieverConnectionError("Qdrant search connection failed.")

    return RetrieverOperationError("Qdrant search failed.")


@lru_cache(maxsize=1)
def get_qdrant_retriever() -> QdrantRetriever:
    """Return a reusable Qdrant retriever configured from environment variables."""
    return QdrantRetriever()
