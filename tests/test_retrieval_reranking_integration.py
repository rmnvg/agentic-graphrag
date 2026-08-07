"""Tests that hybrid candidates are reranked before retrieval responses are built."""

from app.services.embeddings.base_embedder import BaseEmbedder
from app.services.reranking import BaseReranker
from app.services.reranking_service import RerankingService
from app.services.retrieval.base_retriever import BaseRetriever, RetrievedChunk
from app.services.retrieval_service import retrieve_relevant_chunks


class FakeEmbedder(BaseEmbedder):
    """Embedder fake returning one valid query vector."""

    @property
    def model_name(self) -> str:
        """Return a test model name."""
        return "fake"

    @property
    def embedding_dimension(self) -> int:
        """Return a test embedding dimension."""
        return 2

    def embed_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Return a deterministic query vector."""
        assert batch_size == 1
        return [[0.1, 0.2]]


class FakeHybridRetriever(BaseRetriever):
    """Retriever fake that exposes the requested RRF candidate size."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.requested_top_k: int | None = None

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return predefined RRF candidates."""
        self.requested_top_k = top_k
        return self.chunks


class ReverseReranker(BaseReranker):
    """Reranker fake that makes reranking visible in the output order."""

    def rerank(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Reverse the RRF order without modifying any chunk fields."""
        return list(reversed(retrieved_chunks))


def _chunk(chunk_id: str) -> RetrievedChunk:
    """Build a stable retrieved chunk."""
    return RetrievedChunk(
        score=0.1,
        document_id="document-1",
        chunk_id=chunk_id,
        page=1,
        pages=[1],
        section="Introduction",
        text=chunk_id,
        token_count=1,
        metadata={"source": chunk_id},
    )


def test_retrieval_pipeline_requests_candidates_then_returns_reranked_top_k() -> None:
    hybrid_retriever = FakeHybridRetriever([_chunk("rrf-first"), _chunk("rrf-second")])
    reranking_service = RerankingService(ReverseReranker(), max_candidates=20)

    result = retrieve_relevant_chunks(
        "Question",
        top_k=1,
        embedder=FakeEmbedder(),
        retriever=hybrid_retriever,
        reranking_service=reranking_service,
    )

    assert hybrid_retriever.requested_top_k == 20
    assert [match["chunk_id"] for match in result["matches"]] == ["rrf-second"]
    assert result["matches"][0]["metadata"] == {"source": "rrf-second"}
