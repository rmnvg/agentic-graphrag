"""Sentence Transformers cross-encoder implementation for chunk reranking."""

import logging
import os
from functools import lru_cache
from time import perf_counter
from typing import Any

from app.services.reranking.base_reranker import (
    BaseReranker,
    RerankerConfigurationError,
    RerankerOperationError,
)
from app.services.retrieval.base_retriever import RetrievedChunk

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_BATCH_SIZE = 32

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reorder RRF candidates using a cached Sentence Transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        model: Any | None = None,
    ) -> None:
        """Configure a lazy cross-encoder model.

        Args:
            model_name: Hugging Face cross-encoder model identifier.
            batch_size: Number of query-chunk pairs scored per batch.
            model: Optional model injection used by tests.
        """
        if not model_name.strip():
            raise RerankerConfigurationError("Cross-encoder model name must not be empty.")
        if isinstance(batch_size, bool) or batch_size < 1:
            raise RerankerConfigurationError("Reranker batch size must be a positive integer.")

        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any | None = model

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Score query-chunk pairs in a batch and return stable descending order."""
        if not retrieved_chunks:
            return []

        started_at = perf_counter()
        try:
            pairs = [(question, chunk.text) for chunk in retrieved_chunks]
            scores = self._get_model().predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
            normalized_scores = [float(score) for score in scores]
            if len(normalized_scores) != len(retrieved_chunks):
                raise ValueError("Cross-encoder returned an unexpected score count.")

            ranked = sorted(
                enumerate(retrieved_chunks),
                key=lambda item: (-normalized_scores[item[0]], item[0]),
            )
            logger.info(
                "Cross-encoder reranked %d chunks in %.3f seconds (top_scores=%s).",
                len(retrieved_chunks),
                perf_counter() - started_at,
                [round(normalized_scores[index], 4) for index, _ in ranked[:5]],
            )
            return [chunk for _, chunk in ranked]
        except RerankerOperationError:
            raise
        except Exception as exc:
            raise RerankerOperationError("Cross-encoder reranking failed.") from exc

    def _get_model(self) -> Any:
        """Load the cross-encoder only when reranking is first requested."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
                logger.info("Loaded cross-encoder reranker model '%s'.", self._model_name)
            except Exception as exc:
                raise RerankerOperationError("Unable to load cross-encoder model.") from exc
        return self._model


def _get_batch_size() -> int:
    """Read the cross-encoder batch size from environment configuration."""
    raw_value = os.getenv("RERANK_BATCH_SIZE")
    if raw_value is None:
        return DEFAULT_RERANK_BATCH_SIZE

    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise RerankerConfigurationError("RERANK_BATCH_SIZE must be a positive integer.") from exc

    if batch_size < 1:
        raise RerankerConfigurationError("RERANK_BATCH_SIZE must be a positive integer.")
    return batch_size


@lru_cache(maxsize=1)
def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Return the shared lazily loaded cross-encoder reranker."""
    return CrossEncoderReranker(
        model_name=os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL),
        batch_size=_get_batch_size(),
    )
