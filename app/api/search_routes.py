"""HTTP routes for semantic document search."""

from fastapi import APIRouter, HTTPException, status

from app.models.search_models import SearchRequest, SearchResponse
from app.services.retrieval_service import (
    InvalidSearchQueryError,
    SemanticRetrievalError,
    retrieve_relevant_chunks,
)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Return top-ranked indexed chunks for a natural-language query."""
    try:
        search_result = retrieve_relevant_chunks(
            query=request.query,
            top_k=request.top_k,
        )
    except InvalidSearchQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SemanticRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve relevant document chunks.",
        ) from exc

    return SearchResponse(**search_result)
