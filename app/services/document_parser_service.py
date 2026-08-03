import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

UPLOAD_DIRECTORY = Path("data/uploads")
PROCESSED_DIRECTORY = Path("data/processed")
MAX_PARSE_FILE_SIZE_BYTES = 20 * 1024 * 1024


class DocumentParserError(Exception):
    """Base exception for document parsing failures."""


class DocumentNotFoundError(DocumentParserError):
    """Raised when the requested uploaded document cannot be found."""


class DocumentParsingFailedError(DocumentParserError):
    """Raised when the PDF parser cannot parse or serialize the requested document."""


def parse_uploaded_document(document_id: str) -> dict[str, str]:
    """Parse an uploaded PDF and persist RAG-friendly structured JSON output.

    Args:
        document_id: UUID assigned during upload. The source PDF is expected at
            data/uploads/<document_id>.pdf.

    Returns:
        API response payload containing the document id, status, and output path.

    Raises:
        DocumentNotFoundError: If the document id is invalid or the PDF is absent.
        DocumentParsingFailedError: If parsing or JSON persistence fails.
    """
    normalized_document_id = _normalize_document_id(document_id)
    pdf_path = UPLOAD_DIRECTORY / f"{normalized_document_id}.pdf"

    if not pdf_path.is_file():
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    if pdf_path.stat().st_size > MAX_PARSE_FILE_SIZE_BYTES:
        raise DocumentParsingFailedError("PDF exceeds the 20 MB parsing limit.")

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIRECTORY / f"{normalized_document_id}.json"
    temporary_output_path = PROCESSED_DIRECTORY / f"{normalized_document_id}.json.tmp"

    try:
        parsed_payload = _parse_pdf_with_pymupdf4llm(
            document_id=normalized_document_id,
            pdf_path=pdf_path,
        )
        _write_json_atomically(
            output_path=output_path,
            temporary_path=temporary_output_path,
            payload=parsed_payload,
        )
    except DocumentParsingFailedError:
        temporary_output_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary_output_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise DocumentParsingFailedError("Failed to write parsed document output.") from exc

    return {
        "document_id": normalized_document_id,
        "status": "parsed",
        "output_path": output_path.as_posix(),
    }


def _normalize_document_id(document_id: str) -> str:
    """Validate and canonicalize the uploaded document UUID."""
    try:
        return str(UUID(document_id))
    except ValueError as exc:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.") from exc


def _parse_pdf_with_pymupdf4llm(document_id: str, pdf_path: Path) -> dict[str, Any]:
    """Convert a PDF into a RAG-oriented structured payload using PyMuPDF4LLM."""
    try:
        import pymupdf4llm

        markdown = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=False,
            write_images=False,
            embed_images=False,
            show_progress=False,
        )
        page_chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            write_images=False,
            embed_images=False,
            show_progress=False,
        )
        full_text = _extract_plain_text(pymupdf4llm=pymupdf4llm, pdf_path=pdf_path, markdown=markdown)
        parser_json = _extract_parser_json(pymupdf4llm=pymupdf4llm, pdf_path=pdf_path)
        document_metadata = _extract_pdf_metadata(pdf_path)

    except Exception as exc:
        raise DocumentParsingFailedError("Failed to parse the PDF document.") from exc

    normalized_pages = _normalize_page_chunks(page_chunks)
    markdown_blocks = _build_markdown_blocks(markdown)

    return {
        "document_id": document_id,
        "source_path": pdf_path.as_posix(),
        "parsed_at": datetime.now(UTC).isoformat(),
        "parser": {
            "name": "pymupdf4llm",
            "input_format": "pdf",
        },
        "metadata": document_metadata,
        "content": {
            "full_text": full_text,
            "markdown": markdown,
            "hierarchy": {
                "pages": normalized_pages,
                "outline": document_metadata.get("outline", []),
            },
            "blocks": markdown_blocks,
            "headings": [block for block in markdown_blocks if block["type"] == "heading"],
            "paragraphs": [block for block in markdown_blocks if block["type"] == "paragraph"],
            "tables": [block for block in markdown_blocks if block["type"] == "table"],
            "images": document_metadata.get("images", []),
            "parser_json": parser_json,
        },
        "rag": {
            "preferred_source": "markdown",
            "chunking_ready": True,
            "notes": [
                "Use markdown for semantic chunking.",
                "Use hierarchy.pages for page-aware metadata.",
                "Use blocks to keep headings, paragraphs, and tables addressable.",
            ],
        },
    }


def _extract_plain_text(pymupdf4llm: Any, pdf_path: Path, markdown: str) -> str:
    """Extract plain text, falling back to markdown with formatting stripped."""
    if hasattr(pymupdf4llm, "to_text"):
        return pymupdf4llm.to_text(str(pdf_path))

    return _markdown_to_plain_text(markdown)


def _extract_parser_json(pymupdf4llm: Any, pdf_path: Path) -> Any:
    """Return PyMuPDF4LLM's native JSON output when the installed version supports it."""
    if not hasattr(pymupdf4llm, "to_json"):
        return None

    raw_json = pymupdf4llm.to_json(str(pdf_path))

    if isinstance(raw_json, str):
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            return raw_json

    return raw_json


def _extract_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """Extract stable PDF metadata, outline, page sizes, and image references."""
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        with pymupdf.open(pdf_path) as document:
            pages: list[dict[str, Any]] = []
            images: list[dict[str, Any]] = []

            for page_index, page in enumerate(document, start=1):
                page_rect = page.rect
                pages.append(
                    {
                        "page_number": page_index,
                        "width": page_rect.width,
                        "height": page_rect.height,
                    }
                )

                for image_index, image in enumerate(page.get_images(full=True), start=1):
                    images.append(
                        {
                            "page_number": page_index,
                            "image_index": image_index,
                            "xref": image[0],
                            "width": image[2],
                            "height": image[3],
                            "bits_per_component": image[4],
                            "colorspace": image[5],
                            "name": image[7],
                            "filter": image[8],
                        }
                    )

            return {
                "page_count": document.page_count,
                "pdf_metadata": document.metadata,
                "outline": _normalize_outline(document.get_toc(simple=False)),
                "pages": pages,
                "images": images,
            }

    except Exception as exc:
        raise DocumentParsingFailedError("Failed to extract PDF metadata.") from exc


def _normalize_outline(toc: list[list[Any]]) -> list[dict[str, Any]]:
    """Normalize PyMuPDF table-of-contents entries into JSON-friendly records."""
    outline: list[dict[str, Any]] = []

    for entry in toc:
        if len(entry) < 3:
            continue

        outline.append(
            {
                "level": entry[0],
                "title": entry[1],
                "page_number": entry[2],
            }
        )

    return outline


def _normalize_page_chunks(page_chunks: Any) -> list[dict[str, Any]]:
    """Normalize PyMuPDF4LLM page chunks into stable page records."""
    if not isinstance(page_chunks, list):
        return []

    normalized_pages: list[dict[str, Any]] = []

    for page_index, chunk in enumerate(page_chunks, start=1):
        if isinstance(chunk, dict):
            normalized_pages.append(
                {
                    "page_number": chunk.get("metadata", {}).get("page", page_index),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "tables": chunk.get("tables", []),
                    "images": chunk.get("images", []),
                    "graphics": chunk.get("graphics", []),
                }
            )
        else:
            normalized_pages.append(
                {
                    "page_number": page_index,
                    "text": str(chunk),
                    "metadata": {},
                    "tables": [],
                    "images": [],
                    "graphics": [],
                }
            )

    return normalized_pages


def _build_markdown_blocks(markdown: str) -> list[dict[str, Any]]:
    """Build RAG-friendly blocks from Markdown without performing semantic chunking."""
    blocks: list[dict[str, Any]] = []
    pending_paragraph: list[str] = []
    pending_table: list[str] = []

    for line in markdown.splitlines():
        stripped_line = line.strip()

        if not stripped_line:
            _flush_table(blocks=blocks, pending_table=pending_table)
            _flush_paragraph(blocks=blocks, pending_paragraph=pending_paragraph)
            continue

        if _is_markdown_table_line(stripped_line):
            _flush_paragraph(blocks=blocks, pending_paragraph=pending_paragraph)
            pending_table.append(stripped_line)
            continue

        _flush_table(blocks=blocks, pending_table=pending_table)

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped_line)
        if heading_match:
            _flush_paragraph(blocks=blocks, pending_paragraph=pending_paragraph)
            blocks.append(
                {
                    "block_id": f"block-{len(blocks) + 1}",
                    "type": "heading",
                    "level": len(heading_match.group(1)),
                    "text": heading_match.group(2).strip(),
                    "markdown": stripped_line,
                }
            )
            continue

        pending_paragraph.append(stripped_line)

    _flush_table(blocks=blocks, pending_table=pending_table)
    _flush_paragraph(blocks=blocks, pending_paragraph=pending_paragraph)

    return blocks


def _is_markdown_table_line(line: str) -> bool:
    """Return whether a Markdown line looks like part of a table."""
    return line.startswith("|") and line.endswith("|")


def _flush_paragraph(blocks: list[dict[str, Any]], pending_paragraph: list[str]) -> None:
    """Append and clear a pending paragraph block."""
    if not pending_paragraph:
        return

    text = " ".join(pending_paragraph).strip()
    blocks.append(
        {
            "block_id": f"block-{len(blocks) + 1}",
            "type": "paragraph",
            "text": text,
            "markdown": "\n".join(pending_paragraph),
        }
    )
    pending_paragraph.clear()


def _flush_table(blocks: list[dict[str, Any]], pending_table: list[str]) -> None:
    """Append and clear a pending Markdown table block."""
    if not pending_table:
        return

    blocks.append(
        {
            "block_id": f"block-{len(blocks) + 1}",
            "type": "table",
            "text": "\n".join(pending_table),
            "markdown": "\n".join(pending_table),
        }
    )
    pending_table.clear()


def _markdown_to_plain_text(markdown: str) -> str:
    """Convert Markdown to simple plain text for search/index metadata."""
    text = re.sub(r"^#{1,6}\s+", "", markdown, flags=re.MULTILINE)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write_json_atomically(
    output_path: Path,
    temporary_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON to a temporary file before replacing the final output path."""
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(_json_safe(payload), output_file, ensure_ascii=False, indent=2)

    temporary_path.replace(output_path)


def _json_safe(value: Any) -> Any:
    """Convert parser-specific objects into JSON-serializable values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]

    return str(value)
