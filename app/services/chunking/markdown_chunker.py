import re
from typing import Any
from uuid import uuid4

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.services.chunking.base_chunker import BaseChunker

CHUNK_SIZE_CHARS = 2_500
CHUNK_OVERLAP_CHARS = 250


class MarkdownChunker(BaseChunker):
    """Chunk parsed Markdown using LangChain's Markdown-aware splitters.

    The splitter first groups content by Markdown headings so section hierarchy
    is preserved. Large sections are then split with a recursive character
    splitter to keep chunks near the target size without adding tokenizer
    dependencies yet.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_CHARS,
        chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    ) -> None:
        """Create a Markdown chunker.

        Args:
            chunk_size: Target character size, roughly 500-800 tokens.
            chunk_overlap: Character overlap used only when splitting large sections.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, parsed_document: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate RAG-ready chunks from parsed Markdown."""
        document_id = str(parsed_document["document_id"])
        markdown = _extract_markdown(parsed_document)

        if not markdown:
            return []

        section_documents = self._split_markdown_by_headers(markdown)
        chunk_documents = self._split_large_sections(section_documents)
        pages = _extract_document_pages(parsed_document)

        return [
            _build_chunk(
                document_id=document_id,
                chunk_index=index,
                text=document.page_content,
                metadata=dict(document.metadata),
                pages=pages,
            )
            for index, document in enumerate(chunk_documents, start=1)
            if document.page_content.strip()
        ]

    def _split_markdown_by_headers(self, markdown: str) -> list[Any]:
        """Split Markdown into hierarchy-aware section documents using LangChain."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        return splitter.split_text(markdown)

    def _split_large_sections(self, section_documents: list[Any]) -> list[Any]:
        """Split oversized Markdown sections using LangChain recursive splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n# ",
                "\n\n## ",
                "\n\n### ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter.split_documents(section_documents)


def _extract_markdown(parsed_document: dict[str, Any]) -> str:
    """Extract Markdown from supported parsed-document shapes."""
    content = parsed_document.get("content", {})
    return str(content.get("markdown") or parsed_document.get("markdown") or "").strip()


def _build_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
    metadata: dict[str, Any],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a stable chunk payload from a LangChain document."""
    clean_text = text.strip()
    section = _section_from_metadata(metadata)
    matched_pages = _infer_pages_from_text(clean_text, pages)

    return {
        "chunk_id": str(uuid4()),
        "document_id": document_id,
        "page": matched_pages[0] if matched_pages else None,
        "pages": matched_pages,
        "section": section,
        "text": clean_text,
        "token_count": _estimate_token_count(clean_text),
        "metadata": {
            "chunk_index": chunk_index,
            "section_hierarchy": metadata,
            "char_count": len(clean_text),
        },
    }


def _section_from_metadata(metadata: dict[str, Any]) -> str:
    """Build a readable section title from Markdown heading metadata."""
    section_parts = [
        str(metadata[key]).strip()
        for key in sorted(metadata)
        if key.startswith("Header") and str(metadata[key]).strip()
    ]

    return " > ".join(section_parts) if section_parts else "Untitled"


def _extract_document_pages(parsed_document: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract page text records from the parsed document hierarchy."""
    content = parsed_document.get("content", {})
    hierarchy = content.get("hierarchy", {})
    raw_pages = hierarchy.get("pages") or parsed_document.get("pages") or []

    if not isinstance(raw_pages, list):
        return []

    pages: list[dict[str, Any]] = []

    for index, page in enumerate(raw_pages, start=1):
        if not isinstance(page, dict):
            continue

        page_number = page.get("page_number") or page.get("page") or index
        pages.append(
            {
                "page_number": _coerce_page_number(page_number),
                "text": str(page.get("text") or ""),
            }
        )

    return pages


def _infer_pages_from_text(text: str, pages: list[dict[str, Any]]) -> list[int]:
    """Infer chunk page numbers by matching chunk text against parsed page text."""
    if not text or not pages:
        return []

    normalized_chunk = _normalize_text_for_matching(text)

    if not normalized_chunk:
        return []

    chunk_snippets = _matching_snippets(normalized_chunk)
    matched_pages: list[int] = []

    for page in pages:
        page_number = page.get("page_number")
        page_text = _normalize_text_for_matching(str(page.get("text") or ""))

        if not isinstance(page_number, int) or not page_text:
            continue

        if any(snippet in page_text for snippet in chunk_snippets):
            matched_pages.append(page_number)

    return sorted(set(matched_pages))


def _matching_snippets(text: str) -> list[str]:
    """Return stable text snippets useful for page matching."""
    words = text.split()

    if len(words) <= 20:
        return [" ".join(words)]

    return [
        " ".join(words[:30]),
        " ".join(words[len(words) // 2 : len(words) // 2 + 30]),
        " ".join(words[-30:]),
    ]


def _normalize_text_for_matching(text: str) -> str:
    """Normalize whitespace, Markdown markers, and casing for page lookup."""
    normalized_text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    normalized_text = re.sub(r"[*_`|]", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    return normalized_text.strip().lower()


def _coerce_page_number(value: Any) -> int | None:
    """Convert page values to integers when possible."""
    if isinstance(value, int):
        return value

    if isinstance(value, str) and value.isdigit():
        return int(value)

    return None


def _estimate_token_count(text: str) -> int:
    """Estimate token count without adding tokenizer dependencies."""
    if not text:
        return 0

    return max(1, round(len(text) / 4))
