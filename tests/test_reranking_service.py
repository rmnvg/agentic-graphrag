"""Tests for bounded reranking orchestration and RRF fallback behavior."""

from app.services.reranking import BaseReranker, RerankerOperationError
from app.services.reranking_service import RerankingService
from app.services.retrieval.base_retriever import RetrievedChunk


class FailingReranker(BaseReranker):
    """Reranker fake that simulates a provider/model failure."""

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Raise the expected provider-neutral reranking error."""
        raise RerankerOperationError("Model unavailable")


def _chunk(chunk_id: str) -> RetrievedChunk:
    """Build an RRF-ranked chunk for fallback assertions."""
    return RetrievedChunk(
        score=0.1,
        document_id="document-1",
        chunk_id=chunk_id,
        page=1,
        pages=[1],
        section="Introduction",
        text=chunk_id,
        token_count=1,
        metadata={"chunk_id": chunk_id},
    )


def test_reranking_service_falls_back_to_rrf_order_on_failure() -> None:
    chunks = [_chunk("rrf-first"), _chunk("rrf-second")]

    result = RerankingService(FailingReranker(), max_candidates=20).rerank(
        "Question", chunks, final_top_k=5
    )

    assert result == chunks
    assert result[0].metadata == {"chunk_id": "rrf-first"}
