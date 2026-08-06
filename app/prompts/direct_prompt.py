"""Prompt construction for conversational requests that do not need retrieval."""


def build_direct_prompt(question: str) -> str:
    """Build a concise prompt for greeting and courtesy-only requests."""
    return f"""You are a helpful AI assistant.

Respond briefly and naturally to this conversational request. Do not claim to
have searched or read uploaded documents.

Request:
{question}

Response:"""
