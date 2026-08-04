"""Grounded prompt construction for single-turn RAG answers."""

from typing import Any

NO_ANSWER_MESSAGE = "I couldn't find that information in the uploaded documents."


def build_rag_prompt(question: str, matches: list[dict[str, Any]]) -> str:
    """Build a citation-aware prompt from retrieved document chunks.

    ``Source N`` labels correspond to the response ``sources`` list in the
    same order, allowing the model's citations to be mapped back to documents.
    """
    context = _format_context(matches)

    return f"""You are an AI assistant.

Answer ONLY using the supplied context. Do not use outside knowledge.
If the answer is not present in the context, reply exactly:
\"{NO_ANSWER_MESSAGE}\"

When you use information from a source, cite it inline as [Source N].
Do not cite a source that does not support the statement.

Context:
{context}

Question:
{question}

Answer:"""


def _format_context(matches: list[dict[str, Any]]) -> str:
    """Render retrieved chunks in a stable, readable, citation-ready format."""
    if not matches:
        return "No relevant document context was retrieved."

    sources: list[str] = []
    for index, match in enumerate(matches, start=1):
        page = match.get("page")
        section = match.get("section") or "Untitled"
        text = str(match.get("text") or "").strip()

        sources.append(
            f"""[Source {index}]
Document ID: {match.get("document_id", "unknown")}
Page: {page if page is not None else "unknown"}
Section: {section}
Content:
{text}
[/Source {index}]"""
        )

    return "\n\n".join(sources)
