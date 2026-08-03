from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.document_models import DocumentUploadResponse
from app.services.document_service import (
    DocumentUploadError,
    InvalidDocumentUploadError,
    save_uploaded_pdf,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    try:
        upload_result = await save_uploaded_pdf(file)
    except InvalidDocumentUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to upload document.",
        ) from exc

    return DocumentUploadResponse(**upload_result)
