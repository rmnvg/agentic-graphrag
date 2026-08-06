"""Independent LLM-as-a-Judge evaluation for generated RAG answers."""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.prompts.judge_prompt import build_judge_prompt
from app.services.llms import BaseLLM, LLMError, get_llm

logger = logging.getLogger(__name__)


class JudgeEvaluation(BaseModel):
    """Strict structured result returned by the answer judge."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "retry"]
    reason: str


class JudgeService:
    """Evaluate RAG answers without rewriting them or changing their content."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        """Create a judge with an optional LLM injection for testing."""
        self._llm = llm

    def evaluate(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        answer: str,
    ) -> JudgeEvaluation:
        """Evaluate an answer and safely accept it when the judge is unavailable.

        The parser accepts only strict JSON matching ``JudgeEvaluation``. Any
        provider, prompt, or parsing failure intentionally falls back to accept
        so an optional evaluation layer never breaks the user-facing workflow.
        """
        try:
            logger.info("Judge request started.")
            prompt = build_judge_prompt(
                question=question,
                retrieved_chunks=retrieved_chunks,
                answer=answer,
            )
            active_llm = self._llm or get_llm()
            raw_response = active_llm.generate(prompt)
            evaluation = JudgeEvaluation.model_validate_json(raw_response)
            logger.info("Judge response received with decision '%s'.", evaluation.decision)
            return evaluation
        except (LLMError, ValidationError, ValueError, TypeError) as exc:
            logger.exception("Judge evaluation failed; defaulting to accept.")
            return _fallback_evaluation(exc)
        except Exception as exc:
            logger.exception("Unexpected judge failure; defaulting to accept.")
            return _fallback_evaluation(exc)


def _fallback_evaluation(error: Exception) -> JudgeEvaluation:
    """Return a safe evaluation result after an optional judge failure."""
    return JudgeEvaluation(
        decision="accept",
        reason=f"Judge unavailable; accepted without evaluation ({type(error).__name__}).",
    )
