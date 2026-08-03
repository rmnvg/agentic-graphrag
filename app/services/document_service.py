from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
PDF_MIME_TYPE = "application/pdf"
UPLOAD_DIRECTORY = Path("data/uploads")
CHUNK_SIZE_BYTES = 1024 * 1024


class DocumentUploadError(Exception):
    """Base exception for document upload failures."""


class InvalidDocumentUploadError(DocumentUploadError):
    """Raised when the uploaded document does not pass validation."""


async def save_uploaded_pdf(file: UploadFile) -> dict[str, str]:
    if file is None:
        raise InvalidDocumentUploadError("File is required.")

    if not file.filename:
        raise InvalidDocumentUploadError("Uploaded file must include a filename.")

    if Path(file.filename).suffix.lower() != ".pdf":
        raise InvalidDocumentUploadError("Only files with a .pdf extension are allowed.")

    if file.content_type != PDF_MIME_TYPE:
        raise InvalidDocumentUploadError(
            "Only PDF files with MIME type application/pdf are allowed."
        )

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid4())
    stored_filename = f"{document_id}.pdf"
    destination_path = UPLOAD_DIRECTORY / stored_filename
    temporary_path = UPLOAD_DIRECTORY / f"{stored_filename}.tmp"

    total_size = 0

    try:
        with temporary_path.open("wb") as destination:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise InvalidDocumentUploadError("Uploaded file exceeds the 20 MB size limit.")

                destination.write(chunk)

        if total_size == 0:
            raise InvalidDocumentUploadError("Uploaded file cannot be empty.")

        temporary_path.replace(destination_path)

    except InvalidDocumentUploadError:
        temporary_path.unlink(missing_ok=True)
        destination_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        destination_path.unlink(missing_ok=True)
        raise DocumentUploadError("Failed to store uploaded document.") from exc
    finally:
        await file.close()

    return {
        "document_id": document_id,
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "status": "uploaded",
    }
