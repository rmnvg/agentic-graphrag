"""Rank-based fusion for dense and lexical retrieval results."""

from dataclasses import replace

from app.services.retrieval.base_retriever import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Fuse rank lists without comparing incompatible cosine and BM25 scores."""
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            chunks_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )[:top_k]
    return [
        replace(chunks_by_id[chunk_id], score=fused_scores[chunk_id])
        for chunk_id in ordered_chunk_ids
    ]
