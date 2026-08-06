"""Deterministic routing rules for the RAG LangGraph workflow."""

import re
from typing import Literal

GraphRoute = Literal["retrieve", "direct"]

_DOCUMENT_PATTERNS = (
    r"\bpdf\b",
    r"\buploaded\b",
    r"\bdocument(?:s)?\b",
    r"\bpage\s+\d+\b",
    r"\bfile\b",
)
_DIRECT_PATTERNS = (
    r"^(?:hi|hello|hey)(?:\s+there)?[!.?]*$",
    r"^(?:thanks|thank you|thx)[!.?]*$",
    r"^(?:bye|goodbye|see you)[!.?]*$",
)


def route_question(question: str) -> GraphRoute:
    """Return the next graph node for a normalized user question.

    Document references take precedence over conversational phrases. All other
    requests default to retrieval because this endpoint is document-focused.
    """
    normalized_question = " ".join(question.lower().split())

    if any(re.search(pattern, normalized_question) for pattern in _DOCUMENT_PATTERNS):
        return "retrieve"

    if any(re.fullmatch(pattern, normalized_question) for pattern in _DIRECT_PATTERNS):
        return "direct"

    return "retrieve"
