"""Gate B live: the governor asks, decides and cuts, and never touches the stream."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from flyball.config import Settings
from flyball.governor import Governor, GovernorConfig, answer_number, answer_value
from flyball.ledger import Ledger
from flyball.policy import plan_governor
from flyball.proxy import create_app
from flyball.state import build_state


def sse(**delta) -> bytes:
    return b"data: " + json.dumps({"id": "c1", "choices": [{"delta": delta}]}).encode() + b"\n\n"


def reasoning(chars: int, step: int = 100) -> list[bytes]:
    return [sse(reasoning_content="x" * step) for _ in range(chars // step)]


class FakeControl:
    """Scripted answers from the upstream's control endpoint."""

    def __init__(self, probes=(), delay: float = 0.0) -> None:
        self.probes = list(probes)
        self.delay = delay
        self.calls: list[dict] = []

    async def __call__(self, payload: dict) -> dict:
        self.calls.append(payload)
        if self.delay:
            await asyncio.sleep(self.delay)
        if payload["action"] == "probe":
            item = self.probes.pop(0) if self.probes else ("{1", -5.0)
            if isinstance(item, Exception):
                raise item
            content, logprob = item
            return {"success": True, "probe": {"content": content, "logprob": logprob}}
        return {"success": True}

    def actions(self) -> list[str]:
        return [c["action"] for c in self.calls]


def drive(governor: Governor, chunks: list[bytes], pause: float = 0.002) -> Governor:
    async def run() -> None:
        for chunk in chunks:
            governor.feed(chunk)
            await asyncio.sleep(pause)  # a real stream leaves time between tokens
        await governor.close()

    asyncio.run(run())
    return governor


def governor(control: FakeControl, **config) -> Governor:
    return Governor(config=GovernorConfig(**config), control=control, model="m")


# --- deciding ----------------------------------------------------------------


def test_probes_as_the_reasoning_doubles_and_cuts_once_confident() -> None:
    control = FakeControl(probes=[("{17", -2.0), ("{18", -0.05)])
    g = drive(governor(control), reasoning(2000))
    assert control.actions() == ["probe", "probe", "reasoning_end"]
    assert [p["at"] for p in g.probes] == [300, 600]
    assert g.cut["reason"] == "confident" and g.cut["ok"]


def test_the_cap_cuts_a_reasoning_that_never_settles() -> None:
    control = FakeControl(probes=[("{1", -3.0)] * 10)
    g = drive(governor(control), reasoning(3600))
    assert [p["at"] for p in g.probes] == [300, 600, 1200, 2400]
    assert g.cut["reason"] == "cap" and g.cut["at"] >= 3200


def test_a_confident_answer_that_is_not_a_number_does_not_cut() -> None:
    control = FakeControl(probes=[("{Yes", 0.0)] * 10)
    g = drive(governor(control, cap=100_000), reasoning(2000))
    assert g.cut is None
    assert "reasoning_end" not in control.actions()


def test_nothing_happens_once_the_reply_has_started() -> None:
    control = FakeControl(probes=[("{5", 0.0)])
    g = drive(governor(control), reasoning(200) + [sse(content="The answer")] + reasoning(1000))
    assert control.calls == []
    assert g.cut is None


def test_a_failing_probe_stops_the_governor_quietly() -> None:
    control = FakeControl(probes=[RuntimeError("upstream went away")])
    g = drive(governor(control), reasoning(4000))
    assert g.stopped.startswith("probe refused")
    assert g.cut is None
    assert control.actions() == ["probe"]


def test_only_one_control_call_is_ever_in_flight() -> None:
    control = FakeControl(probes=[("{1", -3.0)] * 10, delay=0.05)
    g = drive(governor(control, cap=100_000), reasoning(2500, step=50), pause=0.001)
    # Without the guard there would be one probe per chunk past each threshold.
    assert len(g.probes) <= 4


def test_garbage_in_the_stream_is_ignored() -> None:
    control = FakeControl()
    g = drive(governor(control), [b"data: {not json\n\n", b": keep-alive\n\n"] + reasoning(400))
    assert g.reasoning_chars == 400
    assert not g.stopped


def test_an_anchored_cut_commits_to_the_answer_the_probe_found() -> None:
    control = FakeControl(probes=[("{369", -0.01)])
    g = drive(governor(control, anchor=True), reasoning(500))
    end = next(c for c in control.calls if c["action"] == "reasoning_end")
    assert end["message"] == "\n\nSo the final answer is 369.\n"
    assert g.cut["message"] == "So the final answer is 369."


def test_an_anchored_cap_cut_commits_without_naming_an_answer() -> None:
    control = FakeControl(probes=[("{12", -3.0)] * 10)
    drive(governor(control, anchor=True), reasoning(3600))
    end = next(c for c in control.calls if c["action"] == "reasoning_end")
    assert "12" not in end["message"] and "final answer" in end["message"]


def test_without_anchoring_the_cut_carries_no_message() -> None:
    control = FakeControl(probes=[("{369", -0.01)])
    drive(governor(control, anchor=False), reasoning(500))
    end = next(c for c in control.calls if c["action"] == "reasoning_end")
    assert "message" not in end


def test_option_letters_count_as_answers() -> None:
    assert answer_value("{F") == "F"
    assert answer_value(r"{\text{B") == "B"
    assert answer_value("{(C)") == "C"
    assert answer_value("{Yes") is None
    assert answer_value("{42") == "42"


def test_a_confident_option_letter_cuts() -> None:
    control = FakeControl(probes=[("{F", -0.02)])
    g = drive(governor(control), reasoning(500))
    assert g.cut["reason"] == "confident" and g.probes[0]["answer"] == "F"


def test_answer_numbers_are_read_like_the_research_grader() -> None:
    assert answer_number("{12{,}000") == "12000"
    assert answer_number("{-3") == "-3"
    assert answer_number("{3.50") == "3.5"
    assert answer_number("{\\text{B}") is None


# --- who gets governed ---------------------------------------------------------


MATH = {"stream": True, "messages": [{"role": "user", "content": "How many apples are left if I eat 3 of 7?"}]}


def plan(body: dict = MATH, probe: bool = True, killed: bool = False, **settings):
    values = dict(mode="active", governor=True, ab_treatment_share=1.0)
    values.update(settings)
    return plan_governor(Settings(**values), body, build_state(body), probe, killed)


def test_a_streamed_numeric_question_is_governed() -> None:
    assert plan().govern


def test_a_semicolon_in_prose_does_not_make_a_problem_code() -> None:
    body = {"stream": True, "messages": [{"role": "user", "content": (
        "Harold paints every $h$ th picket;\nTanya paints every $t$ th picket.\n"
        "Find the sum of all such numbers.")}]}
    assert plan(body=body).govern


def test_a_multiple_choice_question_is_governed() -> None:
    body = {"stream": True, "messages": [{"role": "user", "content": (
        "Which style of leadership takes account of others' views?\n\n"
        "A. Task-oriented\nB. Servant\nC. Democratic\nD. Autocratic")}]}
    assert plan(body=body).govern


def test_a_pasted_code_block_is_left_alone() -> None:
    body = {"stream": True, "messages": [{"role": "user", "content": (
        "```python\nx = sum(range(10))\n```\nHow much is x?")}]}
    result = plan(body=body)
    assert not result.govern and "code request" in result.reason


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"settings": {"mode": "shadow"}}, "governor is off"),
        ({"settings": {"governor": False}}, "governor is off"),
        ({"killed": True}, "kill switch"),
        ({"probe": False}, "no probe action"),
        ({"body": dict(MATH, stream=False)}, "not streaming"),
        ({"body": dict(MATH, tools=[{"type": "function"}])}, "tools or images"),
        ({"body": dict(MATH, chat_template_kwargs={"enable_thinking": False})}, "turned reasoning off"),
        ({"body": {"stream": True, "messages": [{"role": "user", "content": "Write a Python function to sort a list."}]}},
         "code request"),
        ({"body": {"stream": True, "messages": [{"role": "user", "content": "Tell me about Rome."}]}},
         "multiple-choice answer"),
    ],
)
def test_everything_else_is_left_alone(change: dict, reason: str) -> None:
    result = plan(
        body=change.get("body", MATH),
        probe=change.get("probe", True),
        killed=change.get("killed", False),
        **change.get("settings", {}),
    )
    assert not result.govern
    assert reason in result.reason


def test_the_control_bucket_is_logged_but_not_governed() -> None:
    result = plan(ab_treatment_share=0.0)
    assert not result.govern and result.would_govern and result.bucket == "control"


# --- through the proxy -----------------------------------------------------------


STREAM = [sse(reasoning_content="y" * 100) for _ in range(12)] + [sse(content="42")] + [b"data: [DONE]\n\n"]


def upstream_with_control(seen: list[httpx.Request]):
    async def chunks():
        for chunk in STREAM:
            await asyncio.sleep(0.01)
            yield chunk

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/chat/completions/control":
            payload = json.loads(request.content)
            if payload["id"] == "flyball-capability-check":
                return httpx.Response(200, json={"success": False, "message": "no active completion for this id"})
            if payload["action"] == "probe":
                return httpx.Response(200, json={"success": True, "probe": {"content": "{42", "logprob": -0.01}})
            return httpx.Response(200, json={"success": True})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, content=chunks(), headers={"content-type": "text/event-stream"})
        return httpx.Response(404)

    return handle


def test_the_proxy_governs_without_touching_the_stream(tmp_path) -> None:
    seen: list[httpx.Request] = []
    settings = Settings(upstream="http://upstream.invalid", mode="active", governor=True,
                        ab_treatment_share=1.0, data_dir=tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_with_control(seen)))
        with client.stream("POST", "/v1/chat/completions", json=MATH) as reply:
            received = b"".join(reply.iter_bytes())

    assert received == b"".join(STREAM), "the client must get the upstream's bytes, unchanged"
    sent = next(r for r in seen if r.url.path == "/v1/chat/completions")
    assert json.loads(sent.content)["reasoning_control"] is True
    actions = [json.loads(r.content)["action"] for r in seen if r.url.path.endswith("/control")]
    assert actions[0] == "probe"  # the capability check
    assert "reasoning_end" in actions
    outcome = Ledger(tmp_path).outcomes.read()[-1]
    assert outcome["governor"]["cut"]["reason"] == "confident"


def test_without_the_probe_action_the_proxy_only_forwards(tmp_path) -> None:
    seen: list[httpx.Request] = []
    handler = upstream_with_control(seen)

    def old_server(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/control"):
            seen.append(request)
            return httpx.Response(400, json={"error": {"message": "unknown control action"}})
        return handler(request)

    settings = Settings(upstream="http://upstream.invalid", mode="active", governor=True,
                        ab_treatment_share=1.0, data_dir=tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(old_server))
        with client.stream("POST", "/v1/chat/completions", json=MATH) as reply:
            received = b"".join(reply.iter_bytes())

    assert received == b"".join(STREAM)
    sent = next(r for r in seen if r.url.path == "/v1/chat/completions")
    assert "reasoning_control" not in json.loads(sent.content)
    assert len([r for r in seen if r.url.path.endswith("/control")]) == 1  # only the check
    decision = Ledger(tmp_path).decisions.read()[-1]
    assert decision["governor"]["reason"] == "the upstream has no probe action"
