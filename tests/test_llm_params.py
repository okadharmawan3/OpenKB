"""``reasoning_effort:`` / ``temperature:`` reach every LLM call OpenKB makes.

The keys are resolved by ``openkb.llm_params.resolve_llm_params`` and forwarded
to the wiki compiler's ``litellm.(a)completion`` calls, the Agents-SDK agents
(via ``ModelSettings.extra_args`` / ``temperature``), and PageIndex's local
indexing (via ``IndexConfig.llm_params``) — on both the CLI path (process-wide
stash) and the REST path (``LlmCredentialBundle``).
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from openkb.agent.compiler import _llm_call, _llm_call_async
from openkb.config import (
    LlmCredentialBundle,
    resolve_credential_bundle,
    resolve_model_settings,
    set_timeout,
)
from openkb.llm_params import (
    LlmParams,
    agent_settings,
    get_llm_params,
    resolve_llm_params,
    set_llm_params,
)

MAX_EFFORT = {"reasoning_effort": "max", "allowed_openai_params": ["reasoning_effort"]}


# --- resolve_llm_params -------------------------------------------------------


def test_absent_keys_resolve_to_defaults():
    assert resolve_llm_params({}) == LlmParams()


def test_reasoning_effort_is_forwarded_verbatim():
    # No fixed value list: the valid set differs per model (xhigh vs max).
    assert resolve_llm_params({"reasoning_effort": " max "}).reasoning_effort == "max"
    assert resolve_llm_params({"reasoning_effort": "xhigh"}).reasoning_effort == "xhigh"


def test_reasoning_effort_mapping_is_forwarded_verbatim():
    # The mapping form reaches the Responses API intact through LiteLLM's
    # openai/responses/* bridge, which drops effort strings it doesn't know.
    params = resolve_llm_params({"reasoning_effort": {"effort": "max"}})
    assert params.reasoning_effort == {"effort": "max"}
    assert params.completion_kwargs()["reasoning_effort"] == {"effort": "max"}


@pytest.mark.parametrize("bad", ["", "   ", 3, True, ["high"], {"summary": "auto"}, {"effort": ""}])
def test_invalid_reasoning_effort_is_ignored_with_warning(bad, caplog):
    with caplog.at_level(logging.WARNING, logger="openkb.llm_params"):
        assert resolve_llm_params({"reasoning_effort": bad}).reasoning_effort is None
    assert "reasoning_effort" in caplog.text


def test_explicit_null_temperature_means_never_send():
    params = resolve_llm_params({"temperature": None})
    assert params.temperature is None
    assert params.temperature_explicit is True


@pytest.mark.parametrize("value", [0, 0.7, 1])
def test_numeric_temperature_is_accepted(value):
    params = resolve_llm_params({"temperature": value})
    assert params.temperature == float(value)
    assert params.temperature_explicit is True


@pytest.mark.parametrize("bad", [True, "0.5", -0.1, float("nan"), float("inf")])
def test_invalid_temperature_keeps_library_defaults(bad, caplog):
    with caplog.at_level(logging.WARNING, logger="openkb.llm_params"):
        params = resolve_llm_params({"reasoning_effort": "low", "temperature": bad})
    assert params == LlmParams(reasoning_effort="low")
    assert "temperature" in caplog.text


# --- kwargs builders ----------------------------------------------------------


def test_completion_kwargs():
    assert LlmParams().completion_kwargs() == {}
    assert LlmParams(reasoning_effort="max").completion_kwargs() == MAX_EFFORT
    assert LlmParams(temperature_explicit=True).completion_kwargs() == {"temperature": None}
    assert LlmParams(temperature=0.2, temperature_explicit=True).completion_kwargs() == {
        "temperature": 0.2
    }


def test_agent_settings_merges_timeout_and_reasoning():
    assert agent_settings(LlmParams(), None) == {"extra_args": None, "temperature": None}
    assert agent_settings(LlmParams(reasoning_effort="max", temperature=0.5), 60.0) == {
        "extra_args": {"timeout": 60.0, **MAX_EFFORT},
        "temperature": 0.5,
    }


def test_litellm_keeps_reasoning_effort_and_omits_temperature_for_unmapped_model():
    """Pins why ``allowed_openai_params`` is sent: for a model family LiteLLM
    doesn't map as a reasoning model, ``drop_params`` (PageIndex's default)
    would otherwise silently drop ``reasoning_effort``; ``temperature=None``
    keeps PageIndex's ``temperature=0`` off the request."""
    from litellm.utils import get_optional_params

    kwargs = {"temperature": 0, "drop_params": True}  # PageIndex's built-in defaults
    kwargs.update(LlmParams("max", None, True).completion_kwargs())
    sent = get_optional_params(model="gpt-6-luna", custom_llm_provider="openai", **kwargs)
    assert sent.get("reasoning_effort") == "max"
    assert "temperature" not in sent


def test_litellm_responses_bridge_forwards_effort_mapping():
    """Pins why the mapping form is accepted: LiteLLM's Responses-API bridge
    maps a mapping straight to ``reasoning`` (a plain ``"max"`` is dropped)."""
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    reasoning = LiteLLMResponsesTransformationHandler()._map_reasoning_effort({"effort": "max"})
    assert reasoning["effort"] == "max"


# --- compiler call sites --------------------------------------------------------


def _fake_response():
    choice = MagicMock()
    choice.message.content = "ok"
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_llm_call_forwards_configured_params():
    set_llm_params(LlmParams("max", None, True))
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-6-luna", [{"role": "user", "content": "hi"}], "step")
    kwargs = completion.call_args.kwargs
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    assert kwargs["temperature"] is None


def test_llm_call_omits_params_when_unset():
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-4o", [{"role": "user", "content": "hi"}], "step")
    kwargs = completion.call_args.kwargs
    assert not {"reasoning_effort", "allowed_openai_params", "temperature"} & kwargs.keys()


def test_llm_call_async_uses_bundle_not_process_stash():
    set_llm_params(LlmParams(reasoning_effort="low"))
    bundle = LlmCredentialBundle(llm_params=LlmParams(reasoning_effort="max"))
    with patch(
        "openkb.agent.compiler.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_fake_response(),
    ) as acompletion:
        asyncio.run(
            _llm_call_async("gpt-6-luna", [{"role": "user", "content": "hi"}], "s", bundle=bundle)
        )
    assert acompletion.call_args.kwargs["reasoning_effort"] == "max"


def test_explicit_call_kwarg_wins_over_config():
    set_llm_params(LlmParams(reasoning_effort="max"))
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-6-luna", [{"role": "user", "content": "hi"}], "s", reasoning_effort="low")
    assert completion.call_args.kwargs["reasoning_effort"] == "low"


# --- agents ---------------------------------------------------------------------


def test_resolve_model_settings_carries_params():
    set_timeout(1200.0)
    set_llm_params(LlmParams("max", 0.3, True))
    settings = resolve_model_settings()
    assert settings["extra_args"] == {"timeout": 1200.0, **MAX_EFFORT}
    assert settings["temperature"] == 0.3


def test_query_agent_reads_bundle_params(tmp_path):
    from openkb.agent.query import build_query_agent

    bundle = LlmCredentialBundle(llm_params=LlmParams(reasoning_effort="max"))
    agent = build_query_agent(str(tmp_path), "gpt-6-luna", bundle=bundle)
    assert agent.model_settings.extra_args == MAX_EFFORT


@pytest.mark.asyncio
async def test_skill_graders_carry_params():
    from types import SimpleNamespace

    from openkb.skill.evaluator import grade_one

    set_llm_params(LlmParams(reasoning_effort="max"))
    captured = {}

    async def fake_runner(agent, *args, **kwargs):
        captured["agent"] = agent
        return SimpleNamespace(final_output="NO-TRIGGER")

    with patch("openkb.skill.evaluator.Runner.run", new=AsyncMock(side_effect=fake_runner)):
        await grade_one("desc", "q?", model="gpt-6-luna")
    assert captured["agent"].model_settings.extra_args == MAX_EFFORT


# --- config sources -------------------------------------------------------------


def _write_config(kb_dir, config: dict) -> None:
    openkb_dir = kb_dir / ".openkb"
    openkb_dir.mkdir(parents=True, exist_ok=True)
    (openkb_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_bundle_reads_params_from_kb_config(tmp_path):
    _write_config(tmp_path, {"reasoning_effort": "max", "temperature": None})
    assert resolve_credential_bundle(tmp_path).llm_params == LlmParams("max", None, True)


def test_setup_llm_key_sets_and_resets_process_stash(tmp_path, monkeypatch):
    from openkb.cli import _KNOWN_PROVIDER_KEYS, _setup_llm_key

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    for key in _KNOWN_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    _write_config(tmp_path, {"model": "gpt-4o-mini", "reasoning_effort": "high"})
    _setup_llm_key(tmp_path)
    assert get_llm_params() == LlmParams(reasoning_effort="high")
    _setup_llm_key(None)
    assert get_llm_params() == LlmParams()
