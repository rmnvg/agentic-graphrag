"""HTTP route for single-turn RAG answer generation."""

from fastapi import APIRouter, HTTPException, status

from app.models.chat_models import ChatRequest, ChatResponse
from app.services.rag_service import (
    InvalidQuestionError,
    RAGServiceError,
    generate_rag_answer,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer one question exclusively from retrieved document context."""
    try:
        rag_result = generate_rag_answer(question=request.question)
    except InvalidQuestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RAGServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate a document-grounded answer.",
        ) from exc

    return ChatResponse(**rag_result)
