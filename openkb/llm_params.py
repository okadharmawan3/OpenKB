"""Per-call LLM generation params from config: ``reasoning_effort`` and ``temperature``.

Both keys are forwarded on every LLM call OpenKB makes — the wiki compiler's
``litellm.(a)completion`` calls, the Agents-SDK agents (query, chat, lint,
skill), and PageIndex's local indexing calls — so one config line controls the
whole run. Like ``extra_headers`` / ``timeout`` in :mod:`openkb.config`, the
resolved value is stashed process-wide by the CLI (``set_llm_params``) and
carried per request on ``LlmCredentialBundle.llm_params`` for the REST server.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmParams:
    """Resolved ``reasoning_effort`` / ``temperature`` settings.

    ``temperature_explicit`` separates an explicit ``temperature: null`` (never
    send a temperature, overriding PageIndex's built-in ``0`` for models that
    reject the parameter) from an absent key (keep each library's default).
    """

    reasoning_effort: str | dict[str, Any] | None = None
    temperature: float | None = None
    temperature_explicit: bool = False

    def completion_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``litellm.completion`` / ``litellm.acompletion``."""
        kwargs = self.reasoning_kwargs()
        if self.temperature_explicit:
            kwargs["temperature"] = self.temperature  # None -> LiteLLM omits it
        return kwargs

    def reasoning_kwargs(self) -> dict[str, Any]:
        """The ``reasoning_effort`` kwargs alone (empty when unset)."""
        effort = self.reasoning_effort
        if effort is None:
            return {}
        return {
            "reasoning_effort": dict(effort) if isinstance(effort, dict) else effort,
            # LiteLLM drops (drop_params) or rejects reasoning_effort for models
            # it doesn't map as reasoning models (e.g. new families like gpt-6-*).
            # Forward it as-is and let the provider validate the value.
            "allowed_openai_params": ["reasoning_effort"],
        }


def resolve_llm_params(config: dict) -> LlmParams:
    """Resolve the optional ``reasoning_effort:`` and ``temperature:`` keys.

    ``reasoning_effort`` is a non-empty string, forwarded verbatim — the valid set
    differs per model and API, so the provider decides. A mapping with an
    ``effort`` key (e.g. ``{effort: max}``) is forwarded verbatim too: LiteLLM's
    Responses-API bridge (``openai/responses/<model>``) silently drops effort
    strings it doesn't know (such as ``max``) but passes a mapping through.
    ``temperature`` is a finite number >= 0, or ``null`` to never send one.
    Invalid values are ignored with a warning, matching the other config keys.
    """
    effort = _resolve_effort(config.get("reasoning_effort"))

    if "temperature" not in config:
        return LlmParams(reasoning_effort=effort)
    temperature = config["temperature"]
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        logger.warning(
            "config: 'temperature' must be a number >= 0, or null to never send it, "
            "got %r — ignoring it.",
            temperature,
        )
        return LlmParams(reasoning_effort=effort)
    return LlmParams(
        reasoning_effort=effort,
        temperature=None if temperature is None else float(temperature),
        temperature_explicit=True,
    )


def _resolve_effort(raw: Any) -> str | dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, dict) and isinstance(raw.get("effort"), str) and raw["effort"].strip():
        return dict(raw)
    logger.warning(
        "config: 'reasoning_effort' must be a non-empty string (e.g. high) or a mapping "
        "with an 'effort' key (e.g. {effort: max}), got %r — ignoring it.",
        raw,
    )
    return None


def agent_settings(params: LlmParams, timeout: float | None) -> dict[str, Any]:
    """``extra_args`` + ``temperature`` kwargs for an Agents-SDK ``ModelSettings``.

    ``ModelSettings`` has no ``timeout`` field, and its ``reasoning`` field only
    accepts OpenAI's effort values (up to ``xhigh``), so both travel in
    ``extra_args`` — the SDK's LiteLLM model forwards them to the completion call.
    """
    extra_args: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
    extra_args.update(params.reasoning_kwargs())
    return {"extra_args": extra_args or None, "temperature": params.temperature}


# Process-wide params, set from config by the CLI (cli._setup_llm_key) and read
# at the call sites via get_llm_params() — mirrors config.set_timeout().
_runtime_llm_params = LlmParams()


def set_llm_params(params: LlmParams) -> None:
    """Set the process-wide LLM params."""
    global _runtime_llm_params
    _runtime_llm_params = params


def get_llm_params() -> LlmParams:
    """Return the process-wide LLM params."""
    return _runtime_llm_params
