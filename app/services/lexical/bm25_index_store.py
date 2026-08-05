"""Durable local storage for BM25-searchable chunk records."""

import json
from threading import RLock
from pathlib import Path
from typing import Any

BM25_DIRECTORY = Path("data/bm25")
BM25_INDEX_PATH = BM25_DIRECTORY / "chunks.json"
BM25_INDEX_VERSION = 1
_INDEX_WRITE_LOCK = RLock()


class BM25IndexStoreError(Exception):
    """Raised when the local BM25 index cannot be read or persisted."""


class BM25IndexStore:
    """Persist chunk records used to build an in-memory BM25 index.

    Source records are persisted rather than a pickled BM25 object, keeping the
    artifact portable, inspectable, and safe to rebuild.
    """

    def __init__(self, index_path: Path = BM25_INDEX_PATH) -> None:
        """Create a store backed by the supplied JSON index path."""
        self._index_path = index_path

    @property
    def index_path(self) -> Path:
        """Return the index artifact path used by this store."""
        return self._index_path

    def load_chunks(self) -> list[dict[str, Any]]:
        """Load all persisted lexical chunk records, or an empty corpus."""
        if not self._index_path.is_file():
            return []

        try:
            with self._index_path.open("r", encoding="utf-8") as input_file:
                payload = json.load(input_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise BM25IndexStoreError("Unable to read the local BM25 index.") from exc

        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(chunks, list) or not all(isinstance(chunk, dict) for chunk in chunks):
            raise BM25IndexStoreError("Local BM25 index contains invalid chunk records.")

        return chunks

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Replace one document's lexical records atomically and idempotently."""
        normalized_chunks = [_normalize_chunk(chunk, document_id) for chunk in chunks]
        temporary_path = self._index_path.with_suffix(".json.tmp")

        with _INDEX_WRITE_LOCK:
            try:
                remaining_chunks = [
                    chunk for chunk in self.load_chunks() if chunk.get("document_id") != document_id
                ]
                payload = {
                    "version": BM25_INDEX_VERSION,
                    "chunks": [*remaining_chunks, *normalized_chunks],
                }
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                with temporary_path.open("w", encoding="utf-8") as output_file:
                    json.dump(payload, output_file, ensure_ascii=False, indent=2)
                temporary_path.replace(self._index_path)
            except OSError as exc:
                temporary_path.unlink(missing_ok=True)
                raise BM25IndexStoreError("Unable to persist the local BM25 index.") from exc


def _normalize_chunk(chunk: dict[str, Any], document_id: str) -> dict[str, Any]:
    """Create a stable BM25 record from an indexed chunk payload."""
    chunk_id = chunk.get("chunk_id")
    text = chunk.get("text")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise BM25IndexStoreError("BM25 chunk records require a chunk_id.")
    if not isinstance(text, str) or not text.strip():
        raise BM25IndexStoreError("BM25 chunk records require non-empty text.")

    pages = chunk.get("pages", [])
    metadata = chunk.get("metadata", {})
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page": chunk.get("page"),
        "pages": pages if isinstance(pages, list) else [],
        "section": chunk.get("section"),
        "text": text.strip(),
        "token_count": chunk.get("token_count"),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }
