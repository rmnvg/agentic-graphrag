"""Prompt construction for retrieval-oriented query rewriting."""


def build_query_rewriter_prompt(
    original_question: str,
    previous_answer: str,
    judge_reason: str,
    retry_count: int,
) -> str:
    """Build a prompt that converts a failed question into a better search query.

    The resulting query is used only to retrieve document chunks. The original
    user question remains the question used when generating the final answer.
    """
    return f"""You rewrite user questions for document retrieval in a RAG system.

Return only one standalone rewritten search query. Do not answer the question.
Do not summarize the previous answer. Do not include an explanation, labels,
quotation marks, markdown, or facts that are not present in the inputs.

Make ambiguous references more precise only when the original question,
previous answer, or judge feedback supplies that context. Preserve the user's
intent and make the query suitable for searching uploaded documents.

Original user question:
{original_question}

Previous generated answer:
{previous_answer}

Judge feedback:
{judge_reason}

Retry attempt:
{retry_count}
"""
