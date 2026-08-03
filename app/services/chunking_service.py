import json
from pathlib import Path
from typing import Any
from uuid import UUID

from app.services.chunking import BaseChunker, MarkdownChunker

PROCESSED_DIRECTORY = Path("data/processed")
CHUNKS_DIRECTORY = Path("data/chunks")


class DocumentChunkingError(Exception):
    """Base exception for document chunking failures."""


class ProcessedDocumentNotFoundError(DocumentChunkingError):
    """Raised when the parsed document JSON cannot be found."""


class DocumentChunkingFailedError(DocumentChunkingError):
    """Raised when chunk generation or persistence fails."""


def chunk_processed_document(
    document_id: str,
    chunker: BaseChunker | None = None,
) -> dict[str, Any]:
    """Generate and persist semantic chunks for a processed document.

    Args:
        document_id: UUID assigned during upload and parsing.
        chunker: Optional chunking strategy. Defaults to MarkdownChunker.

    Returns:
        API response payload containing document id, chunk count, and status.

    Raises:
        ProcessedDocumentNotFoundError: If processed JSON does not exist.
        DocumentChunkingFailedError: If reading, chunking, or writing fails.
    """
    normalized_document_id = _normalize_document_id(document_id)
    processed_path = PROCESSED_DIRECTORY / f"{normalized_document_id}.json"

    if not processed_path.is_file():
        raise ProcessedDocumentNotFoundError(
            f"Processed document '{document_id}' was not found."
        )

    CHUNKS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = CHUNKS_DIRECTORY / f"{normalized_document_id}.json"
    temporary_output_path = CHUNKS_DIRECTORY / f"{normalized_document_id}.json.tmp"
    active_chunker = chunker or MarkdownChunker()

    try:
        parsed_document = _read_json(processed_path)
        chunks = active_chunker.chunk(parsed_document)
        chunk_payload = {
            "document_id": normalized_document_id,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        _write_json_atomically(
            output_path=output_path,
            temporary_path=temporary_output_path,
            payload=chunk_payload,
        )
    except DocumentChunkingFailedError:
        temporary_output_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary_output_path.unlink(missing_ok=True)
        raise DocumentChunkingFailedError("Failed to persist document chunks.") from exc
    except Exception as exc:
        temporary_output_path.unlink(missing_ok=True)
        raise DocumentChunkingFailedError("Failed to chunk processed document.") from exc

    return {
        "document_id": normalized_document_id,
        "chunk_count": len(chunks),
        "status": "chunked",
    }


def _normalize_document_id(document_id: str) -> str:
    """Validate and canonicalize the uploaded document UUID."""
    try:
        return str(UUID(document_id))
    except ValueError as exc:
        raise ProcessedDocumentNotFoundError(
            f"Processed document '{document_id}' was not found."
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    """Read a processed document JSON file."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise DocumentChunkingFailedError("Processed document JSON is invalid.") from exc

    if not isinstance(payload, dict):
        raise DocumentChunkingFailedError("Processed document JSON must be an object.")

    return payload


def _write_json_atomically(
    output_path: Path,
    temporary_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON to a temporary file before replacing the final output path."""
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)

    temporary_path.replace(output_path)
