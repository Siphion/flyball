"""Server adapters: each server spoken to in its own words, behind one interface."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from flyball.governor import BOXED
from flyball.policy import _thinking_off
from flyball.profiles import BONSAI_2, LEGACY, EffortLevel
from flyball.servers import (
    IN_PLACE,
    NONE,
    Discovery,
    LlamaCpp,
    LlamaCppController,
    OpenAICompatible,
    Unrenderable,
)

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


@pytest.mark.parametrize("adapter", [LlamaCpp(), OpenAICompatible()])
@pytest.mark.parametrize(
    "level, fields",
    [
        ("none", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("low", {"reasoning_effort": "low"}),
        ("medium", {"reasoning_effort": "medium"}),
        ("xhigh", {"reasoning_effort": "xhigh"}),
    ],
)
def test_every_bonsai_level_is_rendered_into_request_fields(adapter, level, fields) -> None:
    sent = adapter.render_level(BODY, BONSAI_2.level(level), BONSAI_2)
    assert sent == dict(BODY, **fields)
    assert BODY == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def test_switching_thinking_off_is_byte_identical_to_the_first_version() -> None:
    body = dict(BODY, chat_template_kwargs={"foo": 1})
    for b in (BODY, body):
        rendered = LlamaCpp().render_level(b, LEGACY.off_level(), LEGACY)
        assert json.dumps(rendered) == json.dumps(_thinking_off(b))


def test_rendering_overrides_the_opposite_switch() -> None:
    off = LlamaCpp().render_level(dict(BODY, reasoning_effort="xhigh"), BONSAI_2.level("none"), BONSAI_2)
    assert "reasoning_effort" not in off
    on = LlamaCpp().render_level(dict(BODY, chat_template_kwargs={"enable_thinking": False}),
                                 BONSAI_2.level("low"), BONSAI_2)
    assert on["chat_template_kwargs"] == {"enable_thinking": True} and on["reasoning_effort"] == "low"


def test_a_budget_needs_a_server_that_has_one() -> None:
    level = EffortLevel("short", effort="medium", budget_tokens=512)
    assert LlamaCpp().render_level(BODY, level, BONSAI_2)["reasoning_budget_tokens"] == 512
    with pytest.raises(Unrenderable):
        OpenAICompatible().render_level(BODY, level, BONSAI_2)


def test_only_a_server_with_control_is_armed() -> None:
    assert LlamaCpp().arm(BODY)["reasoning_control"] is True
    assert "reasoning_control" not in OpenAICompatible().arm(BODY)


# --- control -----------------------------------------------------------------


def test_the_llama_cpp_controller_sends_the_first_versions_payloads() -> None:
    sent: list[dict] = []

    async def post(payload: dict) -> dict:
        sent.append(payload)
        if payload["action"] == "probe":
            return {"success": True, "probe": {"content": "{42", "logprob": -0.1}}
        return {"success": True}

    controller = LlamaCppController(post=post, model="m")
    probe = asyncio.run(controller.probe("c1", BOXED))
    end = asyncio.run(controller.end_reasoning("c1", "\n\nSo the final answer is 42.\n"))
    assert probe.ok and probe.content == "{42" and probe.logprob == -0.1 and end.ok
    assert list(sent[0].items()) == [("id", "c1"), ("model", "m"), ("action", "probe"),
                                     ("suffix", BOXED.suffix), ("n_predict", 12), ("stop", "}")]
    assert list(sent[1].items()) == [("id", "c1"), ("action", "reasoning_end"), ("model", "m"),
                                     ("message", "\n\nSo the final answer is 42.\n")]


def test_a_server_without_control_says_so_without_any_request() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request expected")))
    controller = OpenAICompatible().controller(client, "http://x.invalid", {}, "m")
    result = asyncio.run(controller.probe("c1", BOXED))
    assert not result.ok and result.unsupported


# --- discovery ---------------------------------------------------------------


def llama_server(seen: list, probe_action: bool = True):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "model_path": "F:\\models\\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf",
                "model_alias": "bonsai2-27b",
                "chat_template_caps": {"supports_reasoning_effort": True},
            })
        if request.url.path == "/v1/chat/completions/control":
            if probe_action:
                return httpx.Response(200, json={"success": False, "message": "no active completion for this id"})
            return httpx.Response(400, json={"error": {"message": "unknown action"}})
        return httpx.Response(404)

    return handle


def openai_server(seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-8B"}]})
        return httpx.Response(404)

    return handle


def discover(handler, kind: str = "auto", check_probe: bool = True):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    discovery = Discovery(kind)
    return discovery, asyncio.run(discovery.get(client, "http://upstream.invalid", {}, check_probe))


def test_llama_cpp_with_the_probe_action_is_discovered() -> None:
    _, (adapter, caps) = discover(llama_server([]))
    assert isinstance(adapter, LlamaCpp)
    assert caps.kind == "llama.cpp" and caps.model_alias == "bonsai2-27b"
    assert caps.probe == IN_PLACE and caps.end_reasoning == IN_PLACE and caps.effort_field is True


def test_stock_llama_cpp_has_no_probe() -> None:
    _, (adapter, caps) = discover(llama_server([], probe_action=False))
    assert isinstance(adapter, LlamaCpp) and caps.probe == NONE


def test_any_other_server_falls_back_to_the_generic_adapter() -> None:
    seen: list = []
    _, (adapter, caps) = discover(openai_server(seen))
    assert type(adapter) is OpenAICompatible
    assert caps.kind == "openai" and caps.model_id == "Qwen/Qwen3-8B" and caps.probe == NONE


def test_an_upstream_that_wants_a_key_is_reported() -> None:
    _, (_, caps) = discover(lambda request: httpx.Response(401))
    assert caps.auth_required


def test_discovery_is_cached() -> None:
    seen: list = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(llama_server(seen)))
    discovery = Discovery()
    asyncio.run(discovery.get(client, "http://upstream.invalid", {}, True))
    calls = len(seen)
    asyncio.run(discovery.get(client, "http://upstream.invalid", {}, True))
    assert len(seen) == calls


def test_an_unknown_server_kind_is_refused() -> None:
    with pytest.raises(ValueError):
        Discovery("tgi")
