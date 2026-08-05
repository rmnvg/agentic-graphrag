"""Hybrid dense-plus-BM25 retrieval with reciprocal rank fusion."""

import os
from functools import lru_cache

from app.services.lexical import BaseLexicalRetriever, LexicalRetrieverError, get_local_bm25_retriever
from app.services.retrieval.base_retriever import (
    BaseRetriever,
    RetrievedChunk,
    RetrieverError,
    RetrieverOperationError,
)
from app.services.retrieval.qdrant_retriever import get_qdrant_retriever
from app.services.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion

DEFAULT_CANDIDATE_MULTIPLIER = 4


class HybridRetriever(BaseRetriever):
    """Combine semantic Qdrant results with local BM25 results using RRF."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        lexical_retriever: BaseLexicalRetriever,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least one.")

        self._dense_retriever = dense_retriever
        self._lexical_retriever = lexical_retriever
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve dense and lexical candidates, then fuse their rankings."""
        candidate_limit = max(top_k, top_k * self._candidate_multiplier)
        try:
            dense_results = self._dense_retriever.retrieve(
                query=query,
                query_vector=query_vector,
                top_k=candidate_limit,
            )
            lexical_results = self._lexical_retriever.search(query=query, top_k=candidate_limit)
        except (RetrieverError, LexicalRetrieverError) as exc:
            raise RetrieverOperationError("Hybrid retrieval failed.") from exc

        return reciprocal_rank_fusion(
            rankings=[dense_results, lexical_results],
            top_k=top_k,
        )


def _get_candidate_multiplier() -> int:
    """Read the candidate multiplier from environment with a safe default."""
    raw_value = os.getenv("HYBRID_CANDIDATE_MULTIPLIER")
    if raw_value is None:
        return DEFAULT_CANDIDATE_MULTIPLIER

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RetrieverOperationError(
            "HYBRID_CANDIDATE_MULTIPLIER must be a positive integer."
        ) from exc

    if value < 1:
        raise RetrieverOperationError(
            "HYBRID_CANDIDATE_MULTIPLIER must be a positive integer."
        )

    return value


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """Return the default cached hybrid retriever for application requests."""
    return HybridRetriever(
        dense_retriever=get_qdrant_retriever(),
        lexical_retriever=get_local_bm25_retriever(),
        candidate_multiplier=_get_candidate_multiplier(),
    )
