from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.document_models import (
    DocumentChunkResponse,
    DocumentEmbeddingResponse,
    DocumentIndexResponse,
    DocumentParseResponse,
    DocumentUploadResponse,
)
from app.services.chunking_service import (
    DocumentChunkingError,
    ProcessedDocumentNotFoundError,
    chunk_processed_document,
)
from app.services.document_parser_service import (
    DocumentNotFoundError,
    DocumentParserError,
    parse_uploaded_document,
)
from app.services.document_service import (
    DocumentUploadError,
    InvalidDocumentUploadError,
    save_uploaded_pdf,
)
from app.services.embedding_service import (
    ChunkedDocumentNotFoundError,
    DocumentEmbeddingError,
    embed_chunked_document,
)
from app.services.indexing_service import (
    DocumentIndexingError,
    EmbeddedDocumentNotFoundError,
    index_embedded_document,
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


@router.post(
    "/{document_id}/parse",
    response_model=DocumentParseResponse,
    status_code=status.HTTP_200_OK,
)
def parse_document(document_id: str) -> DocumentParseResponse:
    """Parse a previously uploaded PDF into structured Markdown JSON."""
    try:
        parse_result = parse_uploaded_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentParserError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to parse document.",
        ) from exc

    return DocumentParseResponse(**parse_result)


@router.post(
    "/{document_id}/chunk",
    response_model=DocumentChunkResponse,
    status_code=status.HTTP_200_OK,
)
def chunk_document(document_id: str) -> DocumentChunkResponse:
    """Generate semantic Markdown chunks from a parsed document."""
    try:
        chunk_result = chunk_processed_document(document_id)
    except ProcessedDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentChunkingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to chunk document.",
        ) from exc

    return DocumentChunkResponse(**chunk_result)


@router.post(
    "/{document_id}/embed",
    response_model=DocumentEmbeddingResponse,
    status_code=status.HTTP_200_OK,
)
def embed_document(document_id: str) -> DocumentEmbeddingResponse:
    """Generate and persist embeddings for a chunked document."""
    try:
        embedding_result = embed_chunked_document(document_id)
    except ChunkedDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to embed document chunks.",
        ) from exc

    return DocumentEmbeddingResponse(**embedding_result)


@router.post(
    "/{document_id}/index",
    response_model=DocumentIndexResponse,
    status_code=status.HTTP_200_OK,
)
def index_document(document_id: str) -> DocumentIndexResponse:
    """Upsert a document's embedded chunks into Qdrant."""
    try:
        indexing_result = index_embedded_document(document_id)
    except EmbeddedDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentIndexingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to index document vectors.",
        ) from exc

    return DocumentIndexResponse(**indexing_result)
