"""Coordinate bounded cross-encoder reranking with a safe RRF fallback."""

import logging
import os
from functools import lru_cache
from time import perf_counter

from app.services.reranking import BaseReranker, RerankerError, get_cross_encoder_reranker
from app.services.retrieval.base_retriever import RetrievedChunk

DEFAULT_MAX_RERANK_CANDIDATES = 20
DEFAULT_FINAL_RERANKED_CHUNKS = 5

logger = logging.getLogger(__name__)


class RerankingService:
    """Rerank a bounded RRF candidate pool without changing chunk payloads."""

    def __init__(
        self,
        reranker: BaseReranker,
        max_candidates: int,
    ) -> None:
        """Create a reranking coordinator around a provider-neutral reranker."""
        if max_candidates < 1:
            raise ValueError("Maximum rerank candidates must be a positive integer.")

        self._reranker = reranker
        self._max_candidates = max_candidates

    @property
    def max_candidates(self) -> int:
        """Return the maximum number of RRF chunks sent to the cross-encoder."""
        return self._max_candidates

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        final_top_k: int,
    ) -> list[RetrievedChunk]:
        """Return final chunks or retain RRF ordering when scoring fails."""
        candidates = retrieved_chunks[: self._max_candidates]
        logger.info(
            "Reranking started (retrieval_count=%d, candidate_count=%d).",
            len(retrieved_chunks),
            len(candidates),
        )
        if not candidates:
            return []

        started_at = perf_counter()
        try:
            reranked_chunks = self._reranker.rerank(question, candidates)
            logger.info(
                "Reranking completed (reranked_count=%d, latency=%.3f seconds).",
                len(reranked_chunks),
                perf_counter() - started_at,
            )
            return reranked_chunks[:final_top_k]
        except RerankerError:
            logger.exception("Reranking failed; retaining original RRF ordering.")
            return candidates[:final_top_k]
        except Exception:
            logger.exception("Unexpected reranking failure; retaining original RRF ordering.")
            return candidates[:final_top_k]


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive integer setting and fail fast for invalid configuration."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def get_max_rerank_candidates() -> int:
    """Return the configured RRF candidate pool size, defaulting to twenty."""
    return _read_positive_int("RERANK_MAX_CANDIDATES", DEFAULT_MAX_RERANK_CANDIDATES)


def get_final_reranked_chunks() -> int:
    """Return the configured default final RAG chunk count, defaulting to five."""
    final_chunks = _read_positive_int(
        "RERANK_FINAL_CHUNKS", DEFAULT_FINAL_RERANKED_CHUNKS
    )
    max_candidates = get_max_rerank_candidates()
    if final_chunks > max_candidates:
        raise ValueError(
            "RERANK_FINAL_CHUNKS must not exceed RERANK_MAX_CANDIDATES."
        )
    return final_chunks


@lru_cache(maxsize=1)
def get_reranking_service() -> RerankingService:
    """Return the shared reranking coordinator for application requests."""
    return RerankingService(
        reranker=get_cross_encoder_reranker(),
        max_candidates=get_max_rerank_candidates(),
    )
