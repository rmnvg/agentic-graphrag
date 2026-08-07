"""Single-turn, grounded RAG answer generation."""

import logging
from time import perf_counter
from typing import Any, Callable

from app.prompts.rag_prompt import build_rag_prompt
from app.services.llms import BaseLLM, LLMError, get_llm
from app.services.retrieval_service import (
    InvalidSearchQueryError,
    SemanticRetrievalError,
    retrieve_relevant_chunks,
)
from app.services.reranking_service import get_final_reranked_chunks

logger = logging.getLogger(__name__)

DEFAULT_RAG_TOP_K = get_final_reranked_chunks()


class RAGServiceError(Exception):
    """Base exception for RAG answer-generation failures."""


class InvalidQuestionError(RAGServiceError):
    """Raised when a chat question is empty after normalization."""


class RAGRetrievalFailedError(RAGServiceError):
    """Raised when supporting chunks cannot be retrieved."""


class PromptGenerationFailedError(RAGServiceError):
    """Raised when the grounded prompt cannot be constructed."""


class RAGGenerationFailedError(RAGServiceError):
    """Raised when the configured LLM cannot generate an answer."""


def generate_rag_answer(
    question: str,
    llm: BaseLLM | None = None,
    retrieval_function: Callable[[str, int], dict[str, Any]] = retrieve_relevant_chunks,
) -> dict[str, Any]:
    """Retrieve supporting chunks and generate one grounded RAG answer.

    This service is intentionally stateless: every request is independent and
    no conversation history, streaming, or agent orchestration is involved.

    Args:
        question: User question to answer from indexed documents.
        llm: Optional LLM injection for tests or a future provider.
        retrieval_function: Optional retrieval injection for tests.

    Returns:
        The normalized question, grounded answer, and ordered source references.

    Raises:
        InvalidQuestionError: If question is empty.
        RAGRetrievalFailedError: If semantic retrieval fails.
        PromptGenerationFailedError: If prompt formatting fails.
        RAGGenerationFailedError: If the LLM request fails.
    """
    retrieval_result = retrieve_rag_chunks(
        question=question,
        retrieval_function=retrieval_function,
    )
    return generate_answer_from_chunks(
        question=retrieval_result["question"],
        matches=retrieval_result["matches"],
        llm=llm,
    )


def retrieve_rag_chunks(
    question: str,
    retrieval_function: Callable[[str, int], dict[str, Any]] = retrieve_relevant_chunks,
) -> dict[str, Any]:
    """Retrieve and validate chunks needed for one grounded RAG answer."""
    normalized_question = _normalize_question(question)
    logger.info("Incoming RAG question received (length=%d).", len(normalized_question))
    logger.info("RAG retrieval started.")

    try:
        retrieval_result = retrieval_function(normalized_question, DEFAULT_RAG_TOP_K)
        matches = _extract_matches(retrieval_result)
    except (InvalidSearchQueryError, SemanticRetrievalError) as exc:
        logger.exception("RAG retrieval failed.")
        raise RAGRetrievalFailedError("Unable to retrieve supporting document chunks.") from exc
    except Exception as exc:
        logger.exception("RAG retrieval failed.")
        raise RAGRetrievalFailedError("Unable to retrieve supporting document chunks.") from exc

    logger.info("Retrieved %d chunks for RAG generation.", len(matches))

    return {"question": normalized_question, "matches": matches}


def generate_answer_from_chunks(
    question: str,
    matches: list[dict[str, Any]],
    llm: BaseLLM | None = None,
) -> dict[str, Any]:
    """Build a grounded prompt and generate an answer from retrieved chunks."""
    normalized_question = _normalize_question(question)
    logger.info("RAG prompt generation started.")

    try:
        prompt = build_rag_prompt(question=normalized_question, matches=matches)
    except Exception as exc:
        logger.exception("RAG prompt generation failed.")
        raise PromptGenerationFailedError("Unable to build the RAG prompt.") from exc

    logger.info("RAG prompt generation completed. Sending Groq request.")
    request_started_at = perf_counter()

    try:
        active_llm = llm or get_llm()
        answer = active_llm.generate(prompt)
    except LLMError as exc:
        logger.exception("Groq request failed.")
        raise RAGGenerationFailedError("Unable to generate an answer.") from exc
    except Exception as exc:
        logger.exception("Groq request failed.")
        raise RAGGenerationFailedError("Unable to generate an answer.") from exc

    logger.info("Groq response received in %.3f seconds.", perf_counter() - request_started_at)

    return {
        "question": normalized_question,
        "answer": answer,
        "sources": [_source_from_match(match) for match in matches],
    }


def _normalize_question(question: str) -> str:
    """Trim and validate a single-turn RAG question."""
    if not isinstance(question, str):
        raise InvalidQuestionError("Question must be a non-empty string.")

    normalized_question = question.strip()
    if not normalized_question:
        raise InvalidQuestionError("Question must not be empty.")

    return normalized_question


def _extract_matches(retrieval_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the match collection returned by the retrieval service."""
    matches = retrieval_result.get("matches")
    if not isinstance(matches, list):
        raise RAGRetrievalFailedError("Retrieval result must include matches.")

    if not all(isinstance(match, dict) for match in matches):
        raise RAGRetrievalFailedError("Retrieved matches must be objects.")

    return matches


def _source_from_match(match: dict[str, Any]) -> dict[str, Any]:
    """Return the public source reference corresponding to one prompt source."""
    return {
        "document_id": str(match.get("document_id") or ""),
        "page": match.get("page"),
        "section": match.get("section"),
    }
