"""Prompt construction for answer-groundedness evaluation."""

from typing import Any


def build_judge_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: str,
) -> str:
    """Build a strict JSON-only evaluation prompt for the judge service."""
    context = _format_context(retrieved_chunks)

    return f"""You are a strict RAG answer evaluator.

Evaluate the generated answer against the retrieved context only.

Assess all of the following:
1. The answer is grounded in the retrieved context.
2. The answer fully addresses the user's question.
3. The answer contains no unsupported or hallucinated claims.
4. The answer is internally consistent.

Do not rewrite, improve, or answer the question yourself.
Return only one JSON object with exactly these fields:
{{"decision":"accept"|"retry","reason":"short explanation"}}

Retrieved context:
{context}

User question:
{question}

Generated answer:
{answer}
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as evaluation context without adding new facts."""
    if not chunks:
        return "No retrieved document context was provided."

    return "\n\n".join(
        f"""[Chunk {index}]
Document ID: {chunk.get("document_id", "unknown")}
Page: {chunk.get("page", "unknown")}
Section: {chunk.get("section", "Untitled")}
Text:
{str(chunk.get("text") or "").strip()}
[/Chunk {index}]"""
        for index, chunk in enumerate(chunks, start=1)
    )
