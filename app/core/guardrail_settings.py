"""Environment-backed configuration for optional safeguard checks."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

GuardrailFailMode = Literal["open", "closed"]


@dataclass(frozen=True, slots=True)
class GuardrailSettings:
    """Server-side settings controlling input and output safety checks."""

    enabled: bool
    input_enabled: bool
    output_enabled: bool
    input_fail_mode: GuardrailFailMode
    output_fail_mode: GuardrailFailMode
    timeout_seconds: float


def _read_bool(name: str, default: bool) -> bool:
    """Read an explicit boolean environment setting."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"true", "1", "yes", "on"}:
        return True
    if normalized_value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _read_fail_mode(name: str) -> GuardrailFailMode:
    """Read a supported failure policy for one guardrail stage."""
    value = os.getenv(name, "open").strip().lower()
    if value not in {"open", "closed"}:
        raise ValueError(f"{name} must be either 'open' or 'closed'.")
    return value  # type: ignore[return-value]


def _read_timeout() -> float:
    """Read a positive model-call timeout in seconds."""
    raw_value = os.getenv("GUARDRAIL_TIMEOUT_SECONDS", "2")
    try:
        timeout_seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError("GUARDRAIL_TIMEOUT_SECONDS must be a positive number.") from exc

    if timeout_seconds <= 0:
        raise ValueError("GUARDRAIL_TIMEOUT_SECONDS must be a positive number.")
    return timeout_seconds


@lru_cache(maxsize=1)
def get_guardrail_settings() -> GuardrailSettings:
    """Return cached guardrail settings loaded during application startup."""
    return GuardrailSettings(
        enabled=_read_bool("GUARDRAIL_ENABLED", False),
        input_enabled=_read_bool("GUARDRAIL_INPUT_ENABLED", False),
        output_enabled=_read_bool("GUARDRAIL_OUTPUT_ENABLED", False),
        input_fail_mode=_read_fail_mode("GUARDRAIL_INPUT_FAIL_MODE"),
        output_fail_mode=_read_fail_mode("GUARDRAIL_OUTPUT_FAIL_MODE"),
        timeout_seconds=_read_timeout(),
    )
