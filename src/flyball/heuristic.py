"""The heuristic gate: the baseline every learned backend has to beat.

It answers the same questions a learned backend will, in the same shape: a value
and a probability distribution per question. The probabilities here are crude
priors, not calibrated estimates; they exist so that the log, the evaluation
tooling and the calibration layer treat every backend identically.

If a learned backend cannot beat these thirty-odd lines at equal coverage, it has
not earned the right to make decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .state import RequestState

# The decision schema for Gate A. Its hash is stamped on every log row, so that a
# dataset never silently mixes answers to different questions.
SCHEMA = {
    "needs_deliberation": {"type": "noul"},
    "effort": {"type": "score", "levels": ["low", "medium", "xhigh"]},
    "request_class": {
        "type": "choice",
        "options": ["trivial", "format", "lookup", "code", "reasoning", "tool", "vision"],
    },
}
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Decision:
    backend: str
    values: dict
    probabilities: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)


def _classify(s: RequestState) -> tuple[str, float]:
    if s.has_tools:
        return "tool", 0.8
    if s.n_images:
        return "vision", 0.8
    # Several reasoning markers win even in a short message: "why does this
    # deadlock, explain step by step" is brief and still needs thought.
    if s.math_markers or s.reasoning_markers >= 2 or (s.reasoning_markers and s.last_user_chars > 120):
        return "reasoning", 0.6
    if s.code_markers:
        return "code", 0.6
    if s.format_markers:
        return "format", 0.6
    if s.last_user_chars < 80:
        return "trivial", 0.6
    return "lookup", 0.4


def decide(state: RequestState) -> Decision:
    request_class, class_p = _classify(state)

    # Deliberation pays on reasoning-shaped requests and is overhead elsewhere:
    # measured on the exploratory sweep, trivial, format, lookup and tool requests
    # never produce enough reasoning for a cut to matter.
    deliberate_p = {
        "reasoning": 0.8,
        "code": 0.55,
        "lookup": 0.35,
        "vision": 0.3,
        "format": 0.15,
        "tool": 0.15,
        "trivial": 0.1,
    }[request_class]
    if state.client_disabled_thinking:
        deliberate_p = 0.0

    if deliberate_p >= 0.7:
        effort = "xhigh"
    elif deliberate_p >= 0.4:
        effort = "medium"
    else:
        effort = "low"

    others = [o for o in SCHEMA["request_class"]["options"] if o != request_class]
    spread = (1.0 - class_p) / len(others)
    class_dist = {o: round(spread, 4) for o in others}
    class_dist[request_class] = round(class_p, 4)

    return Decision(
        backend="heuristic",
        values={
            "needs_deliberation": deliberate_p >= 0.5,
            "effort": effort,
            "request_class": request_class,
        },
        probabilities={
            "needs_deliberation": {"true": deliberate_p, "false": round(1 - deliberate_p, 4)},
            "request_class": class_dist,
        },
        confidence={
            "needs_deliberation": round(max(deliberate_p, 1 - deliberate_p), 4),
            "request_class": round(class_p, 4),
        },
    )
