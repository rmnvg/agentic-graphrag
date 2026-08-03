import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.services.embeddings import BaseEmbedder, get_sentence_transformer_embedder

CHUNKS_DIRECTORY = Path("data/chunks")
EMBEDDINGS_DIRECTORY = Path("data/embeddings")
PIPELINE_VERSION = "1.0.0"
DEFAULT_EMBEDDING_BATCH_SIZE = 32


class DocumentEmbeddingError(Exception):
    """Base exception for document embedding failures."""


class ChunkedDocumentNotFoundError(DocumentEmbeddingError):
    """Raised when the chunked document JSON cannot be found."""


class DocumentEmbeddingFailedError(DocumentEmbeddingError):
    """Raised when embedding generation or persistence fails."""


def embed_chunked_document(
    document_id: str,
    embedder: BaseEmbedder | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Generate and persist embeddings for all chunks in a chunked document.

    Args:
        document_id: UUID assigned during upload, parsing, and chunking.
        embedder: Optional embedding provider. Defaults to cached Sentence Transformers.
        batch_size: Optional encoding batch size. Defaults to EMBEDDING_BATCH_SIZE
            environment variable, then 32.

    Returns:
        API response payload containing embedding model, dimension, count, and status.

    Raises:
        ChunkedDocumentNotFoundError: If the chunk JSON file does not exist.
        DocumentEmbeddingFailedError: If chunk JSON is invalid, embedding fails, or
            output persistence fails.
    """
    normalized_document_id = _normalize_document_id(document_id)
    chunks_path = CHUNKS_DIRECTORY / f"{normalized_document_id}.json"

    if not chunks_path.is_file():
        raise ChunkedDocumentNotFoundError(
            f"Chunked document '{document_id}' was not found."
        )

    EMBEDDINGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = EMBEDDINGS_DIRECTORY / f"{normalized_document_id}.json"
    temporary_output_path = EMBEDDINGS_DIRECTORY / f"{normalized_document_id}.json.tmp"
    active_batch_size = batch_size or _get_embedding_batch_size()

    try:
        chunk_payload = _read_json(chunks_path)
        chunks = _extract_chunks(chunk_payload)
        texts = [_chunk_text(chunk) for chunk in chunks]
        active_embedder = embedder or get_sentence_transformer_embedder()
        embeddings = active_embedder.embed_texts(texts, batch_size=active_batch_size)

        if len(embeddings) != len(chunks):
            raise DocumentEmbeddingFailedError(
                "Embedding count does not match chunk count."
            )

        embedded_chunks = [
            _build_embedded_chunk(
                document_id=normalized_document_id,
                chunk=chunk,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        embedding_payload = {
            "pipeline_version": PIPELINE_VERSION,
            "document_id": normalized_document_id,
            "embedding_model": active_embedder.model_name,
            "embedding_dimension": active_embedder.embedding_dimension,
            "created_at": datetime.now(UTC).isoformat(),
            "chunks": embedded_chunks,
        }
        _write_json_atomically(
            output_path=output_path,
            temporary_path=temporary_output_path,
            payload=embedding_payload,
        )
    except DocumentEmbeddingFailedError:
        temporary_output_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary_output_path.unlink(missing_ok=True)
        raise DocumentEmbeddingFailedError("Failed to persist embeddings.") from exc
    except Exception as exc:
        temporary_output_path.unlink(missing_ok=True)
        raise DocumentEmbeddingFailedError("Failed to generate embeddings.") from exc

    return {
        "document_id": normalized_document_id,
        "embedding_model": active_embedder.model_name,
        "embedding_dimension": active_embedder.embedding_dimension,
        "embedded_chunks": len(embedded_chunks),
        "status": "embedded",
    }


def _normalize_document_id(document_id: str) -> str:
    """Validate and canonicalize the uploaded document UUID."""
    try:
        return str(UUID(document_id))
    except ValueError as exc:
        raise ChunkedDocumentNotFoundError(
            f"Chunked document '{document_id}' was not found."
        ) from exc


def _get_embedding_batch_size() -> int:
    """Read embedding batch size from environment with a safe default."""
    raw_batch_size = os.getenv("EMBEDDING_BATCH_SIZE")

    if raw_batch_size is None:
        return DEFAULT_EMBEDDING_BATCH_SIZE

    try:
        batch_size = int(raw_batch_size)
    except ValueError as exc:
        raise DocumentEmbeddingFailedError(
            "EMBEDDING_BATCH_SIZE must be a positive integer."
        ) from exc

    if batch_size <= 0:
        raise DocumentEmbeddingFailedError(
            "EMBEDDING_BATCH_SIZE must be a positive integer."
        )

    return batch_size


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise DocumentEmbeddingFailedError("Chunked document JSON is invalid.") from exc

    if not isinstance(payload, dict):
        raise DocumentEmbeddingFailedError("Chunked document JSON must be an object.")

    return payload


def _extract_chunks(chunk_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract chunk objects from a chunk JSON payload."""
    chunks = chunk_payload.get("chunks")

    if not isinstance(chunks, list):
        raise DocumentEmbeddingFailedError("Chunked document JSON must include chunks.")

    normalized_chunks: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise DocumentEmbeddingFailedError("Every chunk must be a JSON object.")

        text = _chunk_text(chunk)

        if not text:
            continue

        normalized_chunk = dict(chunk)
        normalized_chunk.setdefault("chunk_id", f"chunk-{index}")
        normalized_chunks.append(normalized_chunk)

    return normalized_chunks


def _chunk_text(chunk: dict[str, Any]) -> str:
    """Return text used for embedding a chunk."""
    return str(chunk.get("text") or "").strip()


def _build_embedded_chunk(
    document_id: str,
    chunk: dict[str, Any],
    embedding: list[float],
) -> dict[str, Any]:
    """Merge a source chunk with its embedding vector."""
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": document_id,
        "page": chunk.get("page"),
        "pages": chunk.get("pages", []),
        "section": chunk.get("section"),
        "text": _chunk_text(chunk),
        "token_count": chunk.get("token_count"),
        "embedding": embedding,
        "metadata": chunk.get("metadata", {}),
    }


def _write_json_atomically(
    output_path: Path,
    temporary_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON to a temporary file before replacing the final output path."""
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)

    temporary_path.replace(output_path)
