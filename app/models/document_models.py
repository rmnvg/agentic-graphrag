from typing import Literal

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier assigned to the uploaded document.")
    original_filename: str = Field(..., description="Filename provided by the client.")
    stored_filename: str = Field(..., description="Filename used to store the document on disk.")
    status: Literal["uploaded"] = Field(..., description="Upload status.")


class DocumentParseResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier of the parsed document.")
    status: Literal["parsed"] = Field(..., description="Parsing status.")
    output_path: str = Field(..., description="Path to the structured JSON parsing output.")


class DocumentChunkResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier of the chunked document.")
    chunk_count: int = Field(..., ge=0, description="Number of chunks generated for the document.")
    status: Literal["chunked"] = Field(..., description="Chunking status.")
