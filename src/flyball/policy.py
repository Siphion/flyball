"""Policy: turning a decision into an action, or deliberately not.

A decision is only an opinion. This module decides whether to act on it, and every
gate is a reason not to:

- the mode is not `active`;
- the kill switch is set;
- no confidence threshold has been fitted for the question;
- the client already chose for itself;
- the decision would not change anything;
- the confidence is below the fitted threshold;
- the request fell in the control bucket of the A/B split.

Three plans live here:

- `plan`, the first version of Gate A: the heuristic's `needs_deliberation`, only in
  the safe direction, switching thinking *off* for a request judged not to need it;
- `plan_effort`, Gate A learned: a System 1 picks a level of the model's reasoning
  ladder (gate_a.py). When it is on, it replaces the first version;
- `plan_governor`, Gate B: whether the mid-stream governor watches the request.

The A/B split is per conversation, not per request. Measuring per request would
measure agreement with the heuristic; measuring per conversation measures what a
user actually experiences, which is the point. Gate A and Gate B use independent
splits, so that each one's effect can be told apart from the other's.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

from .config import Settings
from .heuristic import Decision
from .state import RequestState


@dataclass
class Plan:
    applied: bool
    would_apply: bool
    reason: str
    bucket: str = ""
    body: bytes | None = None  # rewritten body to send; None means the original


def conversation_bucket(body: dict, treatment_share: float, salt: str = "") -> str:
    """Stable bucket for a conversation: its first user message decides, so every
    turn of the same conversation lands in the same arm. A salt gives another gate
    a split independent of this one."""
    first = ""
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            first = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            break
    digest = hashlib.sha256((salt + first).encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    return "treatment" if position < treatment_share else "control"


def _thinking_off(body: dict) -> dict:
    rewritten = dict(body)
    rewritten["chat_template_kwargs"] = dict(body.get("chat_template_kwargs") or {}, enable_thinking=False)
    return rewritten


def plan(
    settings: Settings,
    body: dict,
    state: RequestState,
    decision: Decision,
    killed: bool,
    render_off: Callable[[dict], dict] = _thinking_off,
) -> Plan:
    if settings.mode != "active":
        return Plan(False, False, "mode is " + settings.mode)
    if killed:
        return Plan(False, False, "kill switch is set")

    threshold = settings.thresholds.get("needs_deliberation")
    if threshold is None:
        return Plan(False, False, "no fitted threshold for needs_deliberation")

    bucket = conversation_bucket(body, settings.ab_treatment_share)

    kwargs = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in kwargs or state.client_effort:
        return Plan(False, False, "the client chose its own reasoning setting", bucket)
    if decision.values.get("needs_deliberation"):
        return Plan(False, False, "decision keeps thinking on; nothing to change", bucket)

    confidence = decision.confidence.get("needs_deliberation", 0.0)
    if confidence < threshold:
        return Plan(False, False, "confidence {:.2f} below threshold {:.2f}".format(confidence, threshold), bucket)

    if bucket == "control":
        return Plan(False, True, "control bucket", bucket)

    # How a request asks for no reasoning depends on the server and the model;
    # the caller passes it in (servers.py, profiles.py).
    rewritten = render_off(body)
    return Plan(True, True, "thinking disabled", bucket, json.dumps(rewritten).encode("utf-8"))


@dataclass
class GovernorPlan:
    govern: bool
    would_govern: bool
    reason: str
    bucket: str = ""

    def as_dict(self) -> dict:
        return {"govern": self.govern, "would_govern": self.would_govern,
                "reason": self.reason, "bucket": self.bucket}


def plan_governor(
    settings: Settings,
    body: dict,
    state: RequestState,
    probe_supported: bool,
    killed: bool,
    level: str = "",
    has_point: bool = True,
) -> GovernorPlan:
    """Whether Gate B governs this request. Every gate is, again, a reason not to.

    The governor asks the model, mid-reasoning, for its final answer in a box, and
    cuts once the model is confident in it. That was measured on questions whose
    answer is a number or an option letter, at given reasoning levels; anything
    else is left alone until it has been measured too. `has_point` says whether the
    model's profile has an operating point for the request's `level`.
    """
    if settings.mode != "active" or not settings.governor:
        return GovernorPlan(False, False, "governor is off")
    if killed:
        return GovernorPlan(False, False, "kill switch is set")
    if not state.stream:
        return GovernorPlan(False, False, "not streaming: there is no mid-stream to govern")
    if state.client_disabled_thinking or state.client_effort == "none":
        return GovernorPlan(False, False, "the client turned reasoning off")
    if state.has_tools or state.n_images:
        return GovernorPlan(False, False, "tools or images: not measured")
    if state.code_request_markers:
        return GovernorPlan(False, False, "a code request: the probe asks for a number")
    if not (state.math_markers or state.numeric_answer_markers or state.choice_options >= 3):
        return GovernorPlan(False, False, "not a question with a numeric or multiple-choice answer")
    if not has_point:
        return GovernorPlan(False, False, "no operating point for level {!r} on this model".format(level))
    if not probe_supported:
        return GovernorPlan(False, False, "the upstream has no probe action")

    bucket = conversation_bucket(body, settings.ab_treatment_share)
    if bucket == "control":
        return GovernorPlan(False, True, "control bucket", bucket)
    return GovernorPlan(True, True, "governed", bucket)


@dataclass
class EffortPlan:
    applied: bool
    would_apply: bool
    reason: str
    level: str = ""  # the level the System 1 chose
    bucket: str = ""
    body: dict | None = None  # the request at that level, when applied

    def as_dict(self) -> dict:
        return {"applied": self.applied, "would_apply": self.would_apply, "reason": self.reason,
                "level": self.level, "bucket": self.bucket}


def plan_effort(settings: Settings, body: dict, profile, adapter, requested: str, choice, killed: bool) -> EffortPlan:
    """Whether Gate A moves this request to the level its System 1 chose.

    `requested` is the level the request runs at without Gate A (profiles.py),
    `choice` the System 1's EffortChoice (gate_a.py), or None when there is no
    usable model. In shadow mode every gate is still checked, so that the log says
    what active mode would have done.
    """
    if killed:
        return EffortPlan(False, False, "kill switch is set")
    if choice is None:
        return EffortPlan(False, False, "no Gate A model for this profile")
    if choice.abstained:
        return EffortPlan(False, False, "the System 1 abstained: " + choice.reason)
    level = choice.level
    names = profile.level_names()
    client_chose = "reasoning_effort" in body or profile.thinking_kwarg in (body.get("chat_template_kwargs") or {})
    if client_chose:
        if settings.gate_a_client_effort == "respect":
            return EffortPlan(False, False, "the client chose its own reasoning level", level)
        if requested not in names or names.index(level) >= names.index(requested):
            return EffortPlan(False, False, "the client's level is no higher than the choice", level)
    if level == requested:
        return EffortPlan(False, False, "the choice is the request's own level", level)
    target = profile.level(level)
    if target is None:
        return EffortPlan(False, False, f"level {level!r} is not on the profile's ladder", level)
    try:
        rewritten = adapter.render_level(body, target, profile)
    except ValueError as exc:
        return EffortPlan(False, False, f"the server cannot express level {level!r}: {exc}", level)
    share = settings.ab_treatment_share if settings.gate_a_share is None else settings.gate_a_share
    bucket = conversation_bucket(body, share, salt="effort:")
    if bucket == "control":
        return EffortPlan(False, True, "control bucket", level, bucket)
    if settings.mode != "active":
        return EffortPlan(False, True, "mode is " + settings.mode, level, bucket)
    return EffortPlan(True, True, f"moved to {level}", level, bucket, rewritten)
