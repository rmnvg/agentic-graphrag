"""Local persisted BM25 implementation for modest document corpora."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.services.lexical.base_lexical_retriever import BaseLexicalRetriever, LexicalRetrieverError
from app.services.lexical.bm25_index_store import BM25IndexStore, BM25IndexStoreError
from app.services.retrieval.base_retriever import RetrievedChunk


class LocalBM25Retriever(BaseLexicalRetriever):
    """Search persisted chunk text with an in-memory, file-aware BM25 cache."""

    def __init__(self, index_store: BM25IndexStore | None = None) -> None:
        self._index_store = index_store or BM25IndexStore()
        self._corpus_fingerprint: tuple[int, int] | None = None
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return keyword matches, rebuilding the cache only when the file changes."""
        query_tokens = tokenize(query)
        if not query_tokens or top_k < 1:
            return []

        self._refresh_index_if_needed()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-float(scores[index]), self._chunks[index]["chunk_id"]),
        )[:top_k]
        return [_to_retrieved_chunk(self._chunks[index], float(scores[index])) for index in ranked_indexes]

    def _refresh_index_if_needed(self) -> None:
        fingerprint = _file_fingerprint(self._index_store.index_path)
        if fingerprint == self._corpus_fingerprint:
            return

        try:
            chunks = self._index_store.load_chunks()
        except BM25IndexStoreError as exc:
            raise LexicalRetrieverError("Unable to load the local BM25 index.") from exc

        self._chunks = chunks
        tokenized_corpus = [tokenize(str(chunk.get("text") or "")) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._corpus_fingerprint = fingerprint


def tokenize(text: str) -> list[str]:
    """Use one predictable normalization for BM25 documents and queries."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _to_retrieved_chunk(chunk: dict[str, Any], score: float) -> RetrievedChunk:
    try:
        return RetrievedChunk(
            score=score,
            document_id=str(chunk["document_id"]),
            chunk_id=str(chunk["chunk_id"]),
            page=_optional_int(chunk.get("page")),
            pages=_page_numbers(chunk.get("pages")),
            section=_optional_string(chunk.get("section")),
            text=str(chunk["text"]),
            token_count=_optional_int(chunk.get("token_count")),
            metadata=chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LexicalRetrieverError("Local BM25 index contains an invalid chunk.") from exc


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_numbers(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [page for item in value if (page := _optional_int(item)) is not None]


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@lru_cache(maxsize=1)
def get_local_bm25_retriever() -> LocalBM25Retriever:
    """Return the shared local BM25 retriever instance."""
    return LocalBM25Retriever()
