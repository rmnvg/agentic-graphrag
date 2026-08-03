"""Index embedded document chunks in a vector store."""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from app.services.vectorstores import (
    BaseVectorStore,
    VectorPoint,
    VectorStoreError,
    get_vectorstore,
)

logger = logging.getLogger(__name__)

EMBEDDINGS_DIRECTORY = Path("data/embeddings")
DOCUMENTS_COLLECTION = "documents"
DEFAULT_UPSERT_BATCH_SIZE = 100


class DocumentIndexingError(Exception):
    """Base exception for document indexing failures."""


class EmbeddedDocumentNotFoundError(DocumentIndexingError):
    """Raised when the generated embedding JSON cannot be found."""


class DocumentIndexingFailedError(DocumentIndexingError):
    """Raised when embedding data cannot be indexed."""


def index_embedded_document(
    document_id: str,
    vector_store: BaseVectorStore | None = None,
    upsert_batch_size: int | None = None,
) -> dict[str, Any]:
    """Upsert every embedded chunk for a document into Qdrant.

    The stable ``chunk_id`` is used as the Qdrant point ID, making retries
    idempotent: an existing chunk is updated rather than duplicated.

    Args:
        document_id: UUID assigned to the document pipeline.
        vector_store: Optional store injection for tests or alternate backends.
        upsert_batch_size: Optional number of points submitted per Qdrant request.

    Returns:
        API response payload with indexing outcome details.

    Raises:
        EmbeddedDocumentNotFoundError: If the embeddings JSON does not exist.
        DocumentIndexingFailedError: If data validation or Qdrant indexing fails.
    """
    normalized_document_id = _normalize_document_id(document_id)
    embeddings_path = EMBEDDINGS_DIRECTORY / f"{normalized_document_id}.json"

    if not embeddings_path.is_file():
        raise EmbeddedDocumentNotFoundError(
            f"Embedded document '{document_id}' was not found."
        )

    try:
        embedding_payload = _read_json(embeddings_path)
        points, vector_size = _build_vector_points(
            embedding_payload=embedding_payload,
            document_id=normalized_document_id,
        )
        active_vector_store = vector_store or get_vectorstore()
        active_batch_size = upsert_batch_size or _get_upsert_batch_size()

        logger.info(
            "Indexing %d vectors for document '%s' into collection '%s'.",
            len(points),
            normalized_document_id,
            DOCUMENTS_COLLECTION,
        )
        active_vector_store.ensure_collection(DOCUMENTS_COLLECTION, vector_size)

        for point_batch in _batches(points, active_batch_size):
            active_vector_store.upsert_points(DOCUMENTS_COLLECTION, point_batch)

        logger.info(
            "Completed indexing %d vectors for document '%s'.",
            len(points),
            normalized_document_id,
        )
    except DocumentIndexingError:
        logger.exception("Document indexing failed for '%s'.", document_id)
        raise
    except VectorStoreError as exc:
        logger.exception("Qdrant indexing failed for '%s'.", document_id)
        raise DocumentIndexingFailedError("Unable to index document vectors.") from exc
    except (OSError, ValueError, TypeError) as exc:
        logger.exception("Invalid embedding data for '%s'.", document_id)
        raise DocumentIndexingFailedError("Unable to index document vectors.") from exc

    return {
        "document_id": normalized_document_id,
        "collection": DOCUMENTS_COLLECTION,
        "indexed_vectors": len(points),
        "status": "indexed",
    }


def _normalize_document_id(document_id: str) -> str:
    """Validate and canonicalize a document UUID."""
    try:
        return str(UUID(document_id))
    except ValueError as exc:
        raise EmbeddedDocumentNotFoundError(
            f"Embedded document '{document_id}' was not found."
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    """Read and validate an embeddings JSON object."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise DocumentIndexingFailedError("Embedding JSON is invalid.") from exc

    if not isinstance(payload, dict):
        raise DocumentIndexingFailedError("Embedding JSON must be an object.")

    return payload


def _build_vector_points(
    embedding_payload: dict[str, Any],
    document_id: str,
) -> tuple[list[VectorPoint], int]:
    """Create validated vector-store points from the embeddings artifact."""
    chunks = embedding_payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise DocumentIndexingFailedError("Embedding JSON must include at least one chunk.")

    points: list[VectorPoint] = []
    vector_size: int | None = None

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise DocumentIndexingFailedError("Every embedded chunk must be an object.")

        point_id = _validate_chunk_id(chunk.get("chunk_id"))
        vector = _validate_vector(chunk.get("embedding"))

        if vector_size is None:
            vector_size = len(vector)
        elif len(vector) != vector_size:
            raise DocumentIndexingFailedError("Embedding vectors must use one dimension.")

        points.append(
            VectorPoint(
                point_id=point_id,
                vector=vector,
                payload={
                    "document_id": document_id,
                    "chunk_id": point_id,
                    "page": chunk.get("page"),
                    "pages": chunk.get("pages", []),
                    "section": chunk.get("section"),
                    "text": str(chunk.get("text") or ""),
                    "token_count": chunk.get("token_count"),
                    "metadata": chunk.get("metadata", {}),
                },
            )
        )

    if vector_size is None:
        raise DocumentIndexingFailedError("Embedding JSON must include at least one vector.")

    declared_dimension = embedding_payload.get("embedding_dimension")
    if declared_dimension is not None and declared_dimension != vector_size:
        raise DocumentIndexingFailedError("Embedding dimension does not match vector data.")

    return points, vector_size


def _validate_chunk_id(value: Any) -> str:
    """Ensure the upstream chunk identifier is a Qdrant-compatible UUID."""
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DocumentIndexingFailedError("Every embedded chunk must have a UUID chunk_id.") from exc


def _validate_vector(value: Any) -> list[float]:
    """Validate one non-empty, numeric embedding vector."""
    if not isinstance(value, list) or not value:
        raise DocumentIndexingFailedError("Every embedded chunk must include a vector.")

    try:
        vector = [float(component) for component in value]
    except (TypeError, ValueError) as exc:
        raise DocumentIndexingFailedError("Embedding vectors must contain numbers.") from exc

    if not all(math.isfinite(component) for component in vector):
        raise DocumentIndexingFailedError("Embedding vectors must contain finite numbers.")

    return vector


def _get_upsert_batch_size() -> int:
    """Read Qdrant upsert batch size from environment with a safe default."""
    raw_batch_size = os.getenv("QDRANT_UPSERT_BATCH_SIZE")
    if raw_batch_size is None:
        return DEFAULT_UPSERT_BATCH_SIZE

    try:
        batch_size = int(raw_batch_size)
    except ValueError as exc:
        raise DocumentIndexingFailedError(
            "QDRANT_UPSERT_BATCH_SIZE must be a positive integer."
        ) from exc

    if batch_size <= 0:
        raise DocumentIndexingFailedError(
            "QDRANT_UPSERT_BATCH_SIZE must be a positive integer."
        )

    return batch_size


def _batches(points: list[VectorPoint], batch_size: int) -> list[list[VectorPoint]]:
    """Split points into bounded upsert requests."""
    return [points[index : index + batch_size] for index in range(0, len(points), batch_size)]
