"""Generate brief conversational responses that do not require retrieval."""

import logging
from time import perf_counter
from typing import Any

from app.prompts.direct_prompt import build_direct_prompt
from app.services.llms import BaseLLM, LLMError, get_llm
from app.services.rag_service import (
    InvalidQuestionError,
    PromptGenerationFailedError,
    RAGGenerationFailedError,
)

logger = logging.getLogger(__name__)


def generate_direct_answer(
    question: str,
    llm: BaseLLM | None = None,
) -> dict[str, Any]:
    """Generate a direct conversational response through the existing LLM adapter."""
    normalized_question = _normalize_question(question)

    try:
        prompt = build_direct_prompt(normalized_question)
    except Exception as exc:
        logger.exception("Direct prompt generation failed.")
        raise PromptGenerationFailedError("Unable to build the direct prompt.") from exc

    logger.info("Direct LLM request started.")
    request_started_at = perf_counter()
    try:
        active_llm = llm or get_llm()
        answer = active_llm.generate(prompt)
    except LLMError as exc:
        logger.exception("Direct LLM request failed.")
        raise RAGGenerationFailedError("Unable to generate a direct answer.") from exc
    except Exception as exc:
        logger.exception("Direct LLM request failed.")
        raise RAGGenerationFailedError("Unable to generate a direct answer.") from exc

    logger.info("Direct LLM response received in %.3f seconds.", perf_counter() - request_started_at)
    return {"question": normalized_question, "answer": answer, "sources": []}


def _normalize_question(question: str) -> str:
    """Validate a direct conversational question independently of graph routing."""
    if not isinstance(question, str) or not question.strip():
        raise InvalidQuestionError("Question must not be empty.")

    return question.strip()
