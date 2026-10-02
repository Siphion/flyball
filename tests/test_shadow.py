"""Shadow mode: decide and record, change nothing.

The central property is asserted, not hoped for: the bytes a client receives in
shadow mode are identical to the bytes it receives in off mode.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from flyball import heuristic
from flyball.config import Settings
from flyball.ledger import Ledger, redact
from flyball.proxy import create_app
from flyball.state import EXCERPT_CHARS, build_state
from flyball.stats import join

SSE = (
    b'data: {"id":"c1","choices":[{"delta":{"reasoning_content":"let me think"}}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{"content":"hel"}}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


async def _stream(payload: bytes):
    # Split the body across chunk boundaries on purpose: the outcome tracker must
    # reassemble SSE lines that arrive in pieces.
    for start in range(0, len(payload), 17):
        yield payload[start : start + 17]


def fake_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/chat/completions":
        return httpx.Response(
            200, content=_stream(SSE), headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(200, content=_stream(b'{"ok": true}'), headers={"content-type": "application/json"})


def make_client(mode: str, data_dir) -> TestClient:
    app = create_app(Settings(upstream="http://upstream.invalid", mode=mode, data_dir=data_dir))
    client = TestClient(app)
    client.__enter__()
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream))
    return client


CHAT = {
    "model": "m",
    "stream": True,
    "messages": [{"role": "user", "content": "Why does this deadlock? Explain step by step."}],
}


def _post(client: TestClient) -> bytes:
    with client.stream("POST", "/v1/chat/completions", json=CHAT) as reply:
        return b"".join(reply.iter_bytes())


def test_shadow_returns_exactly_the_same_bytes_as_off(tmp_path) -> None:
    off = make_client("off", tmp_path / "off")
    shadow = make_client("shadow", tmp_path / "shadow")
    assert _post(shadow) == _post(off) == SSE


def test_off_mode_writes_nothing(tmp_path) -> None:
    client = make_client("off", tmp_path)
    _post(client)
    assert not list(tmp_path.glob("*.jsonl"))


def test_shadow_records_a_decision_and_its_outcome(tmp_path) -> None:
    client = make_client("shadow", tmp_path)
    _post(client)

    rows = join(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["applied"] is False
    assert row["schema_sha256"] == heuristic.SCHEMA_SHA256
    assert row["values"]["request_class"] == "reasoning"

    outcome = row["outcome"]
    assert outcome["completed"] is True
    assert outcome["reasoning_chars"] == len("let me think")
    assert outcome["content_chars"] == len("hello")
    assert outcome["finish_reason"] == "stop"
    assert outcome["ttft_s"] is not None


def test_non_chat_requests_are_not_recorded(tmp_path) -> None:
    client = make_client("shadow", tmp_path)
    client.get("/v1/models")
    assert join(tmp_path) == []


def test_malformed_body_is_still_forwarded(tmp_path) -> None:
    client = make_client("shadow", tmp_path)
    reply = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert reply.status_code == 200
    assert join(tmp_path) == []


def test_reasking_the_same_question_is_flagged(tmp_path) -> None:
    client = make_client("shadow", tmp_path)
    _post(client)
    _post(client)
    first, second = join(tmp_path)
    assert first["reasked"] is True
    assert second["reasked"] is False


# --- StateBuilder ----------------------------------------------------------


def test_state_reads_multimodal_content_and_tools() -> None:
    state = build_state(
        {
            "tools": [{"type": "function"}],
            "messages": [
                {"role": "system", "content": "be terse"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this picture?"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                },
            ],
        }
    )
    assert state.has_tools and state.has_system
    assert state.n_images == 1
    assert state.last_user_excerpt == "what is in this picture?"


def test_state_bounds_the_excerpt() -> None:
    state = build_state({"messages": [{"role": "user", "content": "x" * 10_000}]})
    assert len(state.last_user_excerpt) == EXCERPT_CHARS
    assert state.last_user_chars == 10_000


def test_state_is_deterministic() -> None:
    assert build_state(CHAT) == build_state(json.loads(json.dumps(CHAT)))


# --- heuristic ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tools", "klass", "deliberate"),
    [
        ("Reply with OK", False, "trivial", False),
        ("Convert this JSON to CSV, nothing else: [1]", False, "format", False),
        ("A bag has 3 red balls. What is the probability of two red?", False, "reasoning", True),
        ("What is the weather in Milan?", True, "tool", False),
    ],
)
def test_heuristic_classes(text, tools, klass, deliberate) -> None:
    body = {"messages": [{"role": "user", "content": text}]}
    if tools:
        body["tools"] = [{"type": "function"}]
    decision = heuristic.decide(build_state(body))
    assert decision.values["request_class"] == klass
    assert decision.values["needs_deliberation"] is deliberate


def test_heuristic_respects_a_client_that_disabled_thinking() -> None:
    body = dict(CHAT, chat_template_kwargs={"enable_thinking": False})
    assert heuristic.decide(build_state(body)).values["needs_deliberation"] is False


def test_heuristic_distributions_sum_to_one() -> None:
    decision = heuristic.decide(build_state(CHAT))
    for dist in decision.probabilities.values():
        assert abs(sum(dist.values()) - 1.0) < 1e-3


# --- redaction ---------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "Bearer AuBh6lU5tNN3MCKDUtuCULfkXhyS9t8",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "api_key=hunter2",
    ],
)
def test_secrets_are_redacted(secret) -> None:
    assert secret not in redact("please use " + secret + " for this")


def test_ledger_never_writes_a_secret_to_disk(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.decisions.write({"last_user_excerpt": "token=sk-abcdefghijklmnopqrstuvwxyz123456"})
    assert "sk-abcdef" not in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")


def test_non_streaming_outcome_has_no_ttft(tmp_path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        body = b'{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}'
        return httpx.Response(200, content=_stream(body), headers={"content-type": "application/json"})

    app = create_app(Settings(upstream="http://upstream.invalid", mode="shadow", data_dir=tmp_path))
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        client.post("/v1/chat/completions", json=dict(CHAT, stream=False))
    outcome = join(tmp_path)[0]["outcome"]
    assert outcome["content_chars"] == 2
    assert outcome["ttft_s"] is None, "time to first token is meaningless without streaming"
