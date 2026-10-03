"""Gate A, learned: a System 1 picks the reasoning level, and never holds a request up."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from flyball.calibration import LogisticHead, Platt
from flyball.config import Settings
from flyball.gate_a import STATE_FEATURES, Choosers, EffortChoice, EffortChooser, GateAModel
from flyball.ledger import Ledger
from flyball.policy import plan_effort
from flyball.profiles import BONSAI_2, LEGACY
from flyball.proxy import create_app
from flyball.servers import LlamaCpp, OpenAICompatible
from flyball.state import build_state

WIDTH = len(STATE_FEATURES) + 2
LEVELS = ["none", "low", "medium"]


def head(bias: float) -> LogisticHead:
    return LogisticHead([0.0] * WIDTH, [1.0] * WIDTH, [0.0] * WIDTH, bias)


def model(biases=(-2.0, 3.0, 3.2), delta: float = 0.05, profile: str = "bonsai-2") -> GateAModel:
    """Chances of a right answer about 0.12, 0.95 and 0.96: `low` is close enough."""
    return GateAModel(profile=profile, levels=LEVELS, reference_level="medium", features="state",
                      heads={lv: head(b) for lv, b in zip(LEVELS, biases)},
                      platt={lv: Platt() for lv in LEVELS}, delta=delta)


MATH = {"model": "m", "stream": True,
        "messages": [{"role": "user", "content": "How many apples are left if I eat 3 of 7?"}]}


def choose(chooser: EffortChooser, body: dict = MATH) -> EffortChoice:
    return asyncio.run(chooser.choose(build_state(body), "text"))


# --- the System 1 ------------------------------------------------------------


def test_the_cheapest_level_close_enough_to_the_reference_is_chosen() -> None:
    choice = choose(EffortChooser(model()))
    assert not choice.abstained and choice.level == "low"
    assert choose(EffortChooser(model(delta=0.0))).level == "medium"
    assert choose(EffortChooser(model(biases=(3.5, 3.0, 3.2)))).level == "none"


def test_a_slow_system_1_abstains_within_the_timeout() -> None:
    def slow(states, texts):
        time.sleep(0.5)
        return [[0.0] * WIDTH]

    choice = choose(EffortChooser(model(), timeout_s=0.05, extract=slow))
    assert choice.abstained and "within" in choice.reason
    # The request waited for the timeout, not for the backend (asyncio.run itself
    # waits for the thread at shutdown; a server's event loop does not).
    assert choice.ms < 400


def test_a_failing_system_1_abstains() -> None:
    def broken(states, texts):
        raise RuntimeError("backend down")

    assert choose(EffortChooser(model(), extract=broken)).abstained


def test_an_artifact_round_trips_and_is_checked(tmp_path) -> None:
    path = tmp_path / "gate_a.json"
    model().save(path)
    assert GateAModel.load(path).choose(model().predict([0.0] * WIDTH)) == "low"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = "something-else"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        GateAModel.load(path)


def test_a_model_for_another_profile_is_not_used(tmp_path) -> None:
    path = tmp_path / "gate_a.json"
    model(profile="qwen-thinking").save(path)
    chooser, why = Choosers().for_profile(BONSAI_2, str(path))
    assert chooser is None and "trained for profile" in why
    chooser, why = Choosers().for_profile(BONSAI_2, str(tmp_path / "missing.json"))
    assert chooser is None and "cannot load" in why
    assert Choosers().for_profile(LEGACY)[0] is None


# --- the policy ----------------------------------------------------------------


def plan(body: dict = MATH, choice: EffortChoice | None = None, requested: str = "medium",
         adapter=None, killed: bool = False, **settings):
    values = dict(mode="active", gate_a=True, ab_treatment_share=1.0, thresholds={})
    values.update(settings)
    choice = choice if choice is not None else EffortChoice(level="low", abstained=False)
    return plan_effort(Settings(**values), body, BONSAI_2, adapter or LlamaCpp(), requested, choice, killed)


def test_the_request_is_moved_to_the_chosen_level() -> None:
    result = plan()
    assert result.applied and result.body["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"killed": True}, "kill switch"),
        ({"choice": EffortChoice(reason="no features within 150 ms")}, "abstained"),
        ({"body": dict(MATH, reasoning_effort="xhigh")}, "client chose"),
        ({"requested": "low"}, "request's own level"),
        ({"choice": EffortChoice(level="high", abstained=False)}, "not on the profile's ladder"),
        ({"settings": {"ab_treatment_share": 0.0}}, "control bucket"),
        ({"settings": {"mode": "shadow"}}, "mode is shadow"),
    ],
)
def test_every_gate_is_a_reason_not_to_act(change: dict, reason: str) -> None:
    result = plan(body=change.get("body", MATH), choice=change.get("choice"),
                  requested=change.get("requested", "medium"), killed=change.get("killed", False),
                  **change.get("settings", {}))
    assert not result.applied and reason in result.reason


def test_ceiling_mode_may_only_lower_the_clients_level() -> None:
    lower = plan(body=dict(MATH, reasoning_effort="xhigh"), requested="xhigh", gate_a_client_effort="ceiling")
    assert lower.applied and lower.body["reasoning_effort"] == "low"
    higher = plan(body=dict(MATH, reasoning_effort="none"), requested="none", gate_a_client_effort="ceiling")
    assert not higher.applied


def test_a_level_the_server_cannot_express_is_not_applied() -> None:
    budget = BONSAI_2.__class__(**{**BONSAI_2.__dict__, "levels": BONSAI_2.levels + (
        BONSAI_2.levels[2].__class__("short", effort="medium", budget_tokens=256),)})
    result = plan_effort(Settings(mode="active", gate_a=True, ab_treatment_share=1.0, thresholds={}), MATH, budget,
                         OpenAICompatible(), "medium", EffortChoice(level="short", abstained=False), False)
    assert not result.applied and "cannot express" in result.reason


# --- through the proxy -----------------------------------------------------------


SSE = b'data: {"id":"c1","choices":[{"delta":{"content":"4"}}]}\n\ndata: [DONE]\n\n'


async def _stream(payload: bytes):
    yield payload


def upstream(seen: list, reject_rewrites: bool = False):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            seen.append(body)
            if reject_rewrites and "reasoning_effort" in body:
                return httpx.Response(500, content=_stream(b'{"error": "unknown effort"}'))
            return httpx.Response(200, content=_stream(SSE), headers={"content-type": "text/event-stream"})
        return httpx.Response(404, content=_stream(b"{}"))

    return handle


def run(tmp_path, mode: str, reject: bool = False) -> tuple[list, bytes]:
    path = tmp_path / "gate_a.json"
    model().save(path)
    seen: list = []
    settings = Settings(upstream="http://upstream.invalid", mode=mode, data_dir=tmp_path, thresholds={},
                        gate_a=True, gate_a_artifact=str(path), profile="bonsai-2", default_level="medium",
                        ab_treatment_share=1.0)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream(seen, reject)))
        with client.stream("POST", "/v1/chat/completions", json=MATH) as reply:
            received = b"".join(reply.iter_bytes())
    return seen, received


def test_active_mode_sends_the_chosen_level(tmp_path) -> None:
    seen, received = run(tmp_path, "active")
    assert seen[-1]["reasoning_effort"] == "low" and received == SSE
    decision = Ledger(tmp_path).decisions.read()[-1]
    assert decision["gate_a"]["applied"] and decision["level"] == "low"
    assert decision["requested_level"] == "medium"


def test_shadow_mode_sends_the_clients_request_and_logs_the_choice(tmp_path) -> None:
    seen, received = run(tmp_path, "shadow")
    assert seen[-1] == MATH and received == SSE
    outcome = Ledger(tmp_path).outcomes.read()[-1]
    assert outcome["gate_a"]["would_apply"] and outcome["gate_a"]["level"] == "low"


def test_a_rewrite_the_upstream_rejects_is_replaced_by_the_original(tmp_path) -> None:
    seen, received = run(tmp_path, "active", reject=True)
    assert [("reasoning_effort" in body) for body in seen] == [True, False]
    assert received == SSE
    assert Ledger(tmp_path).outcomes.read()[-1]["rewrite_rejected"] == 500


def test_stats_report_what_gate_a_did(tmp_path, capsys) -> None:
    from flyball.stats import report

    run(tmp_path, "active")
    assert report(tmp_path) == 0
    assert "Gate A (learned): 1 requests" in capsys.readouterr().out
