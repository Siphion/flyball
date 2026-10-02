"""Passthrough guarantees, checked against a fake upstream.

These tests must keep passing for every future version of flyball: whatever the
governor learns to do, a request it does not understand has to arrive at the model
unchanged, and an upstream that disappears must not look like a model error.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from flyball.config import Settings
from flyball.proxy import create_app

SEEN: list[httpx.Request] = []


async def _stream(payload: bytes):
    """MockTransport responses must be real streams: the proxy reads them raw."""
    yield payload


def _streamed(status: int, payload: bytes, content_type: str) -> httpx.Response:
    return httpx.Response(
        status, content=_stream(payload), headers={"content-type": content_type}
    )


def fake_upstream(request: httpx.Request) -> httpx.Response:
    SEEN.append(request)
    if request.url.path == "/health":
        return _streamed(200, b'{"status": "ok"}', "application/json")
    if request.url.path == "/v1/models":
        return _streamed(200, b'{"data": [{"id": "fake-model"}]}', "application/json")
    if request.url.path == "/boom":
        return _streamed(503, b'{"error": "upstream said no"}', "application/json")
    if request.url.path == "/v1/chat/completions":
        body = b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n"
        body += b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n"
        body += b"data: [DONE]\n\n"
        return _streamed(200, body, "text/event-stream")
    return _streamed(404, b'{"error": "not found"}', "application/json")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    SEEN.clear()
    settings = Settings(upstream="http://upstream.invalid", mode="off")
    app = create_app(settings)
    with TestClient(app) as test_client:
        # Replace the real outgoing client once startup has created it.
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream))
        yield test_client


def test_get_is_forwarded_and_body_returned(client: TestClient) -> None:
    reply = client.get("/v1/models")
    assert reply.status_code == 200
    assert reply.json() == {"data": [{"id": "fake-model"}]}
    assert SEEN[-1].url.path == "/v1/models"


def test_request_body_and_method_survive(client: TestClient) -> None:
    client.post("/v1/chat/completions", json={"model": "x", "tools": [{"type": "function"}]})
    sent = SEEN[-1]
    assert sent.method == "POST"
    assert b'"tools"' in sent.content, "tool definitions must reach the model untouched"


def test_query_string_survives(client: TestClient) -> None:
    client.get("/v1/models", params={"verbose": "1"})
    assert SEEN[-1].url.params.get("verbose") == "1"


def test_streaming_chunks_are_preserved(client: TestClient) -> None:
    with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as reply:
        payload = b"".join(reply.iter_bytes())
    assert payload.count(b"data: ") == 3
    assert payload.rstrip().endswith(b"data: [DONE]")


def test_upstream_error_status_is_not_rewritten(client: TestClient) -> None:
    assert client.get("/boom").status_code == 503


def test_unreachable_upstream_is_reported_as_such(client: TestClient) -> None:
    app = client.app

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    reply = client.get("/v1/models")
    assert reply.status_code == 502
    assert reply.json()["error"]["type"] == "upstream_unavailable"


def test_upstream_key_replaces_the_client_key(client: TestClient) -> None:
    app = client.app
    app.state.settings.upstream_api_key = "server-side-secret"
    client.get("/v1/models", headers={"Authorization": "Bearer client-token"})
    assert SEEN[-1].headers["authorization"] == "Bearer server-side-secret"


def test_own_health_does_not_touch_the_model(client: TestClient) -> None:
    body = client.get("/flyball/health").json()
    assert body["mode"] == "off"
    assert body["upstream_reachable"] is True


# --- client authentication -------------------------------------------------


@pytest.fixture
def guarded():
    SEEN.clear()
    settings = Settings(
        upstream="http://upstream.invalid",
        mode="off",
        api_key="client-secret",
        upstream_api_key="server-secret",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream))
        yield test_client


def test_missing_key_is_refused_before_reaching_the_model(guarded: TestClient) -> None:
    reply = guarded.get("/v1/models")
    assert reply.status_code == 401
    assert reply.json()["error"]["type"] == "authentication_error"
    assert SEEN == [], "an unauthenticated request must never reach the upstream"


def test_wrong_key_is_refused(guarded: TestClient) -> None:
    reply = guarded.get("/v1/models", headers={"Authorization": "Bearer guess"})
    assert reply.status_code == 401
    assert SEEN == []


def test_right_key_passes_and_is_swapped_for_the_upstream_key(guarded: TestClient) -> None:
    reply = guarded.get("/v1/models", headers={"Authorization": "Bearer client-secret"})
    assert reply.status_code == 200
    assert SEEN[-1].headers["authorization"] == "Bearer server-secret"


def test_x_api_key_header_is_accepted(guarded: TestClient) -> None:
    assert guarded.get("/v1/models", headers={"x-api-key": "client-secret"}).status_code == 200


def test_liveness_and_preflight_stay_public(guarded: TestClient) -> None:
    assert guarded.get("/health").status_code == 200
    assert guarded.get("/flyball/health").status_code == 200
    assert guarded.options("/v1/chat/completions").status_code != 401
