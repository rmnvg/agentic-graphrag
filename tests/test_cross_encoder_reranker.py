"""Tests for cross-encoder ordering without downloading a real model."""

from typing import Any

from app.services.reranking import CrossEncoderReranker
from app.services.retrieval.base_retriever import RetrievedChunk


class FakeCrossEncoder:
    """Deterministic fake cross-encoder that returns configured scores."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        """Record scored pairs and return deterministic relevance scores."""
        self.pairs = pairs
        assert batch_size == 32
        assert show_progress_bar is False
        return self.scores


def _chunk(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    """Build a retrieved chunk with metadata used to verify preservation."""
    return RetrievedChunk(
        score=0.1,
        document_id="document-1",
        chunk_id=chunk_id,
        page=page,
        pages=[page],
        section="Introduction",
        text=text,
        token_count=10,
        metadata={"chunk_index": page},
    )


def test_cross_encoder_reorders_chunks_and_preserves_metadata() -> None:
    first = _chunk("first", "First candidate", 1)
    second = _chunk("second", "Most relevant candidate", 2)
    model = FakeCrossEncoder([0.2, 0.9])

    result = CrossEncoderReranker(model=model).rerank("Question", [first, second])

    assert result == [second, first]
    assert result[0].metadata == {"chunk_index": 2}
    assert result[0].pages == [2]
    assert model.pairs == [
        ("Question", "First candidate"),
        ("Question", "Most relevant candidate"),
    ]


def test_cross_encoder_handles_fewer_than_candidate_limit() -> None:
    only_chunk = _chunk("only", "Only candidate", 1)

    result = CrossEncoderReranker(model=FakeCrossEncoder([0.7])).rerank(
        "Question", [only_chunk]
    )

    assert result == [only_chunk]
