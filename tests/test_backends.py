"""Decision backends speak one protocol, whichever model is behind it."""

from __future__ import annotations

import json

import httpx
import pytest

from flyball.backends import LayaLocal, SystemOneHTTP, backend_from_spec, noul

QUESTIONS = {"settled": {"type": "noul", "instructions": "Has the reasoning reached its answer?"}}


def fake_server(seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json={"answers": {"settled": {"noul": 0.9}}, "usage": {}})
        if request.url.path == "/v1/systemone/batch":
            results = [{"answers": {"settled": {"noul": 0.1 * (i + 1)}}} for i in range(len(body["states"]))]
            return httpx.Response(200, json={"results": results, "total_usage": {}})
        return httpx.Response(404)

    return handle


def backend(seen: list, **kwargs) -> SystemOneHTTP:
    b = SystemOneHTTP("http://systemone.invalid", **kwargs)
    headers = {"Authorization": f"Bearer {b.api_key}"} if b.api_key else {}
    b._client = httpx.Client(transport=httpx.MockTransport(fake_server(seen)), headers=headers)
    return b


def test_one_state_uses_the_single_endpoint() -> None:
    seen: list = []
    answers = backend(seen).decide(["the answer is 4"], QUESTIONS)
    assert noul(answers[0], "settled") == 0.9
    assert seen[0].url.path == "/v1/systemone"
    assert json.loads(seen[0].content)["state"] == {"body": "the answer is 4"}


def test_many_states_are_batched_in_order() -> None:
    seen: list = []
    answers = backend(seen, batch_size=2).decide(["a", "b", "c"], QUESTIONS)
    assert [round(noul(a, "settled"), 1) for a in answers] == [0.1, 0.2, 0.9]
    assert [r.url.path for r in seen] == ["/v1/systemone/batch", "/v1/systemone"]


def test_the_key_and_the_model_are_sent() -> None:
    seen: list = []
    backend(seen, api_key="k", model="typed-decisions").decide(["x"], QUESTIONS)
    assert seen[0].headers["authorization"] == "Bearer k"
    assert json.loads(seen[0].content)["model"] == "typed-decisions"


def test_specs_pick_the_backend() -> None:
    assert isinstance(backend_from_spec("laya"), LayaLocal)
    assert backend_from_spec("laya:multilingual").model == "multilingual"
    assert isinstance(backend_from_spec("https://example.invalid"), SystemOneHTTP)
    with pytest.raises(ValueError):
        backend_from_spec("ollama")
