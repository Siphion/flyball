"""Gate A, learned: a System 1 picks how much the model should think.

Before the call, from the request alone, a System 1 estimates for each level of the
model's reasoning ladder the probability that the model answers correctly at that
level. The cheapest level whose estimate is within `delta` of the reference level's
is chosen: on Bonsai 2 that means stepping down from `medium` when thinking less
is not expected to cost accuracy (docs/gate-b-research.md, section 12).

Every piece is swappable:

- the features: flyball's own request features (`state`), the vector of any
  embedder (`embed:<spec>`, see backends.py), or the answers of any System One
  backend to a few typed questions (`systemone:<spec>`: Laya, Jev, a server);
- the heads: one small logistic model per level, trained by tools/gate_a_eval.py,
  each with its own calibration fitted on held-out data;
- the policy: how much estimated accuracy may be given up for a cheaper level.

All of it comes from one JSON artifact, which names the profile and the ladder it
was trained for; a mismatch disables Gate A rather than mis-steering a model.
Choosing never raises: any failure, or a backend slower than the timeout, is an
abstention, and the request goes out as the client sent it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .backends import ask, backend_from_spec, embedder_from_spec
from .calibration import LogisticHead, Platt, logit
from .state import REQUEST_TEXT_CHARS, RequestState

ARTIFACT_FORMAT = "flyball.gate_a.v1"

# flyball's request features, as numbers. Counts of characters are also given on a
# log scale, where a linear model can use them.
STATE_FEATURES = [
    "n_messages", "n_user_turns", "has_system", "has_tools", "n_images", "last_user_chars",
    "total_chars", "code_markers", "code_request_markers", "math_markers", "numeric_answer_markers",
    "choice_options", "reasoning_markers", "format_markers", "client_disabled_thinking",
]


def state_vector(state: RequestState) -> list[float]:
    values = state.features()
    vector = [float(values[name]) for name in STATE_FEATURES]
    vector += [math.log1p(state.last_user_chars), math.log1p(state.total_chars)]
    return vector


# What a System One backend is asked about a request. Yes-or-no questions only:
# every implementation of the protocol answers those the same way.
QUESTIONS = {
    "needs_steps": {"type": "noul", "instructions":
                    "Does answering this request correctly require working through several steps of reasoning?"},
    "hard": {"type": "noul", "instructions":
             "Is this a hard problem, one that a strong student could easily get wrong?"},
    "quick": {"type": "noul", "instructions":
              "Can this be answered correctly at a glance, without working anything out?"},
}


def answers_vector(answers: dict) -> list[float]:
    """Each question's probability and its logit; 0.5 when a backend skipped one."""
    vector = []
    for name in QUESTIONS:
        try:
            p = float(answers[name]["noul"])
        except (KeyError, TypeError, ValueError):
            p = 0.5
        vector += [p, logit(p)]
    return vector


Extractor = Callable[[list[RequestState], list[str]], list[list[float]]]


def extractor(spec: str) -> Extractor:
    """Features for a batch of requests: `state`, `embed:<embedder spec>`, or
    `systemone:<backend spec>`."""
    kind, _, target = spec.partition(":")
    if kind == "state":
        return lambda states, texts: [state_vector(s) for s in states]
    if kind == "embed":
        embedder = embedder_from_spec(target)
        return lambda states, texts: embedder.embed(texts)
    if kind == "systemone":
        backend = backend_from_spec(target)
        return lambda states, texts: [answers_vector(a) for a in backend.decide(list(texts), QUESTIONS)]
    raise ValueError(f"unknown Gate A feature spec: {spec!r}")


@dataclass
class GateAModel:
    profile: str
    levels: list[str]  # the ladder it chooses from, cheapest first
    reference_level: str  # the level it steps down from
    features: str  # an extractor spec
    heads: dict[str, LogisticHead]
    platt: dict[str, Platt]
    delta: float  # estimated accuracy that may be given up for a cheaper level
    max_chars: int = REQUEST_TEXT_CHARS
    metrics: dict = field(default_factory=dict)

    def predict(self, vector: list[float]) -> dict[str, float]:
        return {level: self.platt[level].apply(self.heads[level].logit(vector)) for level in self.levels}

    def choose(self, p: dict[str, float]) -> str:
        floor = p[self.reference_level] - self.delta
        return next((level for level in self.levels if p[level] >= floor), self.reference_level)

    def as_dict(self) -> dict:
        return {
            "format": ARTIFACT_FORMAT,
            "profile": self.profile,
            "levels": self.levels,
            "reference_level": self.reference_level,
            "features": self.features,
            "max_chars": self.max_chars,
            "heads": {level: head.as_dict() for level, head in self.heads.items()},
            "platt": {level: platt.as_dict() for level, platt in self.platt.items()},
            "policy": {"rule": "relative", "delta": self.delta},
            "metrics": self.metrics,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "GateAModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"{path} is not a Gate A artifact ({ARTIFACT_FORMAT})")
        levels = list(data["levels"])
        if data["reference_level"] not in levels or set(data["heads"]) != set(levels) \
                or set(data["platt"]) != set(levels):
            raise ValueError(f"{path}: heads, calibrations and ladder do not agree")
        return cls(
            profile=data["profile"],
            levels=levels,
            reference_level=data["reference_level"],
            features=data["features"],
            heads={level: LogisticHead.from_dict(h) for level, h in data["heads"].items()},
            platt={level: Platt.from_dict(p) for level, p in data["platt"].items()},
            delta=float(data["policy"]["delta"]),
            max_chars=int(data.get("max_chars", REQUEST_TEXT_CHARS)),
            metrics=data.get("metrics", {}),
        )


@dataclass
class EffortChoice:
    level: str = ""
    p: dict = field(default_factory=dict)
    abstained: bool = True
    reason: str = ""
    ms: float = 0.0

    def as_dict(self) -> dict:
        return {"level": self.level, "p": {k: round(v, 4) for k, v in self.p.items()},
                "abstained": self.abstained, "reason": self.reason, "ms": self.ms}


class EffortChooser:
    """Runs one Gate A model on live requests, within a time budget."""

    def __init__(self, model: GateAModel, timeout_s: float = 0.15, extract: Extractor | None = None) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self._extract = extract or extractor(model.features)

    def compatible_with(self, profile) -> str:
        """Why this model cannot steer `profile`, or "" if it can."""
        if self.model.profile != profile.id:
            return f"trained for profile {self.model.profile!r}, not {profile.id!r}"
        missing = [level for level in self.model.levels if profile.level(level) is None]
        if missing:
            return f"levels {missing} are not on the profile's ladder"
        return ""

    def _vector(self, state: RequestState, text: str) -> list[float]:
        return self._extract([state], [text])[0]

    async def choose(self, state: RequestState, text: str) -> EffortChoice:
        started = time.perf_counter()
        vector = await ask(self._vector, state, text, timeout=self.timeout_s)
        ms = round((time.perf_counter() - started) * 1000, 1)
        if vector is None:
            return EffortChoice(reason="no features within {:.0f} ms".format(self.timeout_s * 1000), ms=ms)
        try:
            p = self.model.predict(vector)
            level = self.model.choose(p)
        except Exception as exc:  # noqa: BLE001 - a broken model abstains, it never breaks a request
            return EffortChoice(reason=f"the model failed: {exc!r}", ms=ms)
        return EffortChoice(level=level, p=p, abstained=False, ms=ms)


class Choosers:
    """Gate A models by artifact path, loaded on first use.

    A model that fails to load, or was trained for another profile, is reported in
    the log of every request it would have decided, and Gate A stays off for it."""

    def __init__(self, timeout_s: float = 0.15) -> None:
        self.timeout_s = timeout_s
        self._cache: dict[str, EffortChooser | str] = {}

    def for_profile(self, profile, artifact: str = "") -> tuple[EffortChooser | None, str]:
        """The chooser for `profile` (an override path wins over the profile's own),
        or None and the reason there is none."""
        path = artifact or profile.gate_a
        if not path:
            return None, "no Gate A model for profile " + profile.id
        if path not in self._cache:
            try:
                self._cache[path] = EffortChooser(GateAModel.load(path), self.timeout_s)
            except Exception as exc:  # noqa: BLE001 - a broken artifact turns Gate A off, nothing else
                self._cache[path] = f"cannot load {path}: {exc!r}"
        chooser = self._cache[path]
        if isinstance(chooser, str):
            return None, chooser
        why = chooser.compatible_with(profile)
        return (None, why) if why else (chooser, "")

    def add(self, path: str, chooser: EffortChooser) -> None:
        self._cache[path] = chooser
