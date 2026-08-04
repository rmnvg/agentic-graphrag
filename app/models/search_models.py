"""Pydantic models for semantic search requests and responses."""

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for semantic chunk retrieval."""

    query: str = Field(..., description="Natural-language question to search for.")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum chunks to return.")


class SearchMatch(BaseModel):
    """A semantically relevant chunk returned by Qdrant."""

    score: float = Field(..., description="Cosine similarity score assigned by Qdrant.")
    document_id: str = Field(..., description="Source document identifier.")
    chunk_id: str = Field(..., description="Source chunk identifier.")
    page: int | None = Field(default=None, description="Primary source page when available.")
    pages: list[int] = Field(default_factory=list, description="All source pages represented by the chunk.")
    section: str | None = Field(default=None, description="Document section title when available.")
    text: str = Field(..., description="Retrieved chunk text.")
    token_count: int | None = Field(default=None, description="Estimated chunk token count.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata.")


class SearchResponse(BaseModel):
    """Response returned after semantic chunk retrieval."""

    query: str = Field(..., description="Normalized query used for retrieval.")
    matches: list[SearchMatch] = Field(..., description="Chunks ordered by relevance score.")
