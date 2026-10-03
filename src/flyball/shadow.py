"""Shadow observation: decide, record, and watch what happens, without acting.

Everything in this module runs beside the passthrough, never in it. The proxy keeps
forwarding the original bytes; this module only reads a copy. Any exception raised
here is caught by the caller and logged, so a bug in observation can cost a log row
but never a response.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from typing import Awaitable, Callable

from . import heuristic
from .config import Settings
from .gate_a import Choosers
from .ledger import Ledger
from .policy import GovernorPlan, Plan, plan, plan_effort, plan_governor
from .profiles import LEGACY, ModelProfile, resolve
from .servers import LlamaCpp, OpenAICompatible, ServerCapabilities
from .state import REQUEST_TEXT_CHARS, STATE_VERSION, build_state, request_text
from .streams import Delta, SSEReader, read_message


@dataclass
class OutcomeTracker:
    """Reads a copy of the upstream response and summarises how it went."""

    decision_id: str
    started: float
    streaming: bool
    status: int = 0
    first_content_at: float | None = None
    reasoning_chars: int = 0
    content_chars: int = 0
    tool_calls: int = 0
    finish_reason: str = ""
    saw_done: bool = False
    unreadable: bool = False
    # Gate B's plan for this request; the proxy starts a governor when it says so.
    governor_plan: GovernorPlan | None = None
    # Who is behind the request, for the governor: the server adapter, the model's
    # profile and the reasoning level the request runs at.
    adapter: OpenAICompatible | None = None
    profile: ModelProfile = LEGACY
    level: str = ""
    # In shadow mode Gate A decides after the request has left; the proxy runs this
    # and records what it would have done in the outcome row.
    gate_a_job: Callable[[], Awaitable[dict]] | None = None
    _reader: SSEReader = field(default_factory=SSEReader)
    _body: list = field(default_factory=list)

    def feed(self, chunk: bytes) -> None:
        if self.unreadable:
            return
        if not self.streaming:
            self._body.append(chunk)
            return
        for delta in self._reader.feed(chunk):
            self._delta(delta)
        self.saw_done = self._reader.done

    def _delta(self, delta: Delta) -> None:
        if delta.reasoning:
            self.reasoning_chars += len(delta.reasoning)
        if delta.content:
            if self.first_content_at is None:
                self.first_content_at = time.time()
            self.content_chars += len(delta.content)
        if delta.tool_calls:
            if self.first_content_at is None:
                self.first_content_at = time.time()
            self.tool_calls += delta.tool_calls
        if delta.finish_reason:
            self.finish_reason = delta.finish_reason

    def finish(self, completed: bool) -> dict:
        if not self.streaming and self._body:
            message = read_message(b"".join(self._body))
            if message is not None:
                self._delta(message)
                self.saw_done = True
        ended = time.time()
        return {
            "decision_id": self.decision_id,
            "ts": round(ended, 3),
            "status": self.status,
            "completed": completed and (self.saw_done or not self.streaming),
            "aborted_by_client": not completed,
            "seconds": round(ended - self.started, 3),
            "ttft_s": round(self.first_content_at - self.started, 3)
            if self.first_content_at and self.streaming
            else None,
            "reasoning_chars": self.reasoning_chars,
            "content_chars": self.content_chars,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
        }


async def observe(
    ledger: Ledger,
    settings: Settings,
    raw_body: bytes,
    probe_supported: bool = False,
    adapter: OpenAICompatible | None = None,
    caps: ServerCapabilities | None = None,
    profiles: list[ModelProfile] | None = None,
    choosers: Choosers | None = None,
) -> tuple[OutcomeTracker | None, bytes]:
    """Decide on a chat completion request, record the decision, and return a
    tracker plus the body to send: the original, unless active mode rewrote it.

    `adapter` and `caps` describe the upstream, as discovered (servers.py); without
    them the upstream is taken to be llama.cpp, as in flyball's first version.
    `profiles` are the model profiles to match the request against, and `choosers`
    the Gate A models (gate_a.py)."""
    try:
        body = json.loads(raw_body)
    except ValueError:
        return None, raw_body
    if not isinstance(body, dict) or not body.get("messages"):
        return None, raw_body

    adapter = adapter if adapter is not None else LlamaCpp()
    caps = caps if caps is not None else ServerCapabilities()
    profile = resolve(caps.model_names() + [str(body.get("model") or "")], profiles or [], settings.profile)
    requested = profile.requested_level(body, settings.default_level)

    state = build_state(body)
    started = time.time()
    decision = heuristic.decide(state)
    decided_in = time.time() - started
    killed = settings.kill_switch.exists()

    effort = None
    gate_a_row = None
    gate_a_job = None
    if settings.gate_a and choosers is not None:
        # The learned Gate A replaces the first version's thinking switch.
        action = Plan(False, False, "Gate A (learned) decides the reasoning level")
        chooser, why = choosers.for_profile(profile, settings.gate_a_artifact)
        text = request_text(body, chooser.model.max_chars if chooser else REQUEST_TEXT_CHARS)

        async def decide_effort():
            choice = await chooser.choose(state, text) if chooser else None
            result = plan_effort(settings, body, profile, adapter, requested, choice, killed)
            row = result.as_dict()
            row["choice"] = choice.as_dict() if choice else None
            row["model"] = chooser.model.features if chooser else why
            return row, result

        async def shadow_job() -> dict:
            row, _ = await decide_effort()
            return row

        if settings.mode == "active":
            gate_a_row, effort = await decide_effort()
        else:
            gate_a_job = shadow_job
    else:
        action = plan(settings, body, state, decision, killed=killed,
                      render_off=lambda b: adapter.render_level(b, profile.off_level(), profile))

    # The request as it will run: moved by Gate A, or with thinking switched off by
    # the first version, or as the client sent it.
    if effort is not None and effort.applied:
        current, level = effort.body, effort.level
    elif action.applied and action.body:
        current, level = json.loads(action.body), profile.off_level().name
    else:
        current, level = body, requested

    target = profile.level(level)
    if (target is not None and not target.thinking) or (action.applied and level == profile.off_level().name):
        # Gate A switched thinking off: there is no reasoning left for Gate B.
        governor_plan = GovernorPlan(False, False, "thinking was switched off by Gate A")
    else:
        governor_plan = plan_governor(settings, current, state, probe_supported, killed=killed,
                                      level=level, has_point=profile.operating_point(level) is not None)

    to_send = raw_body
    if governor_plan.govern:
        # Arm mid-stream control for this completion, so that it can be ended.
        to_send = json.dumps(adapter.arm(current)).encode("utf-8")
    elif current is not body:
        to_send = action.body if action.applied and action.body else json.dumps(current).encode("utf-8")

    decision_id = uuid.uuid4().hex
    ledger.decisions.write(
        {
            "decision_id": decision_id,
            "ts": round(started, 3),
            "mode": settings.mode,
            "applied": action.applied,
            "would_apply": action.would_apply,
            "policy_reason": action.reason,
            "bucket": action.bucket,
            "model": body.get("model", ""),
            "server": caps.kind,
            "profile": profile.id,
            "requested_level": requested,
            "level": level,
            "gate_a": gate_a_row,
            "schema_sha256": heuristic.SCHEMA_SHA256,
            "state_version": STATE_VERSION,
            "backend": decision.backend,
            "decide_ms": round(decided_in * 1000, 3),
            "values": decision.values,
            "probabilities": decision.probabilities,
            "confidence": decision.confidence,
            "features": state.features(),
            "governor": governor_plan.as_dict(),
            "last_user_hash": state.last_user_hash,
            "last_user_excerpt": state.last_user_excerpt,
        }
    )
    tracker = OutcomeTracker(decision_id=decision_id, started=started, streaming=state.stream,
                             governor_plan=governor_plan, adapter=adapter, profile=profile, level=level,
                             gate_a_job=gate_a_job)
    return tracker, to_send
