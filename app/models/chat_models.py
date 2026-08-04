"""Pydantic models for single-turn RAG chat."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for a grounded RAG answer."""

    question: str = Field(..., description="Question to answer from uploaded documents.")


class ChatSource(BaseModel):
    """Document reference corresponding to a cited retrieved source."""

    document_id: str = Field(..., description="Source document identifier.")
    page: int | None = Field(default=None, description="Source page when available.")
    section: str | None = Field(default=None, description="Source section when available.")


class ChatResponse(BaseModel):
    """Response returned after grounded single-turn RAG generation."""

    question: str = Field(..., description="Normalized question used by the RAG pipeline.")
    answer: str = Field(..., description="Answer grounded in retrieved document context.")
    sources: list[ChatSource] = Field(..., description="Sources in the same order as prompt citations.")
