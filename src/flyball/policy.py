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

Only the first question of Gate A is wired here, `needs_deliberation`, and only in
the safe direction: switching thinking *off* for a request judged not to need it.
That direction is reversible and costs time, not correctness, when it is wrong.

The A/B split is per conversation, not per request. Measuring per request would
measure agreement with the heuristic; measuring per conversation measures what a
user actually experiences, which is the point.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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


def conversation_bucket(body: dict, treatment_share: float) -> str:
    """Stable bucket for a conversation: its first user message decides, so every
    turn of the same conversation lands in the same arm."""
    first = ""
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            first = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            break
    digest = hashlib.sha256(first.encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    return "treatment" if position < treatment_share else "control"


def plan(
    settings: Settings,
    body: dict,
    state: RequestState,
    decision: Decision,
    killed: bool,
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

    rewritten = dict(body)
    rewritten["chat_template_kwargs"] = dict(kwargs, enable_thinking=False)
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
) -> GovernorPlan:
    """Whether Gate B governs this request. Every gate is, again, a reason not to.

    The governor asks the model, mid-reasoning, for its final answer in a box, and
    cuts once the model is confident in it. That was measured on questions whose
    answer is a number; anything else is left alone until it has been measured too.
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
    if not (state.math_markers or state.numeric_answer_markers):
        return GovernorPlan(False, False, "not a question with a numeric answer")
    if not probe_supported:
        return GovernorPlan(False, False, "the upstream has no probe action")

    bucket = conversation_bucket(body, settings.ab_treatment_share)
    if bucket == "control":
        return GovernorPlan(False, True, "control bucket", bucket)
    return GovernorPlan(True, True, "governed", bucket)
