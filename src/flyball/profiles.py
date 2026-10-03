"""Model profiles: what flyball needs to know about the model behind the server.

A server adapter (servers.py) knows how to say things to one kind of server. A
profile knows what one model understands:

- its ladder of reasoning levels, cheapest first, in abstract terms: thinking on or
  off, an effort label its chat template knows, an optional token budget. The
  server adapter turns a level into request fields;
- the level a request gets when it names none;
- how its reasoning is written (the tags), and how to ask it for its answer
  mid-reasoning (the probe protocol);
- the Gate B operating points measured for it, level by level;
- a trained Gate A, if there is one.

Operating points are per model and per level because that is how they behave: the
governor that pays at `xhigh` on Bonsai 2 does not pay at `medium`
(docs/gate-b-research.md). A level with no operating point is not governed.

A profile is matched on the model id the server reports, then on the one the
request names. Built-in profiles cover the models measured so far. More can be
loaded from a JSON file (FLYBALL_PROFILES); those take precedence, and one can
extend a built-in to change only what it lists.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Mapping

from .governor import BOXED, GovernorConfig, ProbeProtocol


@dataclass(frozen=True)
class EffortLevel:
    """One rung of a model's reasoning ladder, independent of any server."""

    name: str
    thinking: bool = True
    effort: str = ""  # the reasoning-effort label the template understands; "" sends none
    budget_tokens: int = 0  # a per-request reasoning budget, where the server has one; 0 = none

    def as_dict(self) -> dict:
        return {"name": self.name, "thinking": self.thinking, "effort": self.effort,
                "budget_tokens": self.budget_tokens}


@dataclass(frozen=True)
class ModelProfile:
    id: str
    match: tuple[str, ...] = ()  # case-insensitive globs on the model id
    levels: tuple[EffortLevel, ...] = ()  # cheapest first
    default_level: str = ""  # the level of a request that names none; "" = unknown
    thinking_kwarg: str = "enable_thinking"  # the chat-template switch, a property of the template
    reasoning_tags: tuple[str, str] = ("<think>", "</think>")
    template_opens_reasoning: bool = True  # the prompt already ends inside the reasoning
    probe: ProbeProtocol | None = BOXED
    governor: Mapping[str, GovernorConfig] = field(default_factory=dict)  # per level; "*" = any level
    gate_a: str = ""  # path of a trained Gate A artifact
    notes: str = ""

    def matches(self, model_id: str) -> bool:
        model_id = model_id.lower()
        return any(fnmatch.fnmatchcase(model_id, pattern.lower()) for pattern in self.match)

    def level(self, name: str) -> EffortLevel | None:
        return next((level for level in self.levels if level.name == name), None)

    def level_names(self) -> list[str]:
        return [level.name for level in self.levels]

    def off_level(self) -> EffortLevel:
        """The level that switches reasoning off; every template that has a switch
        has one, whether the profile lists it or not."""
        return next((level for level in self.levels if not level.thinking), EffortLevel("none", thinking=False))

    def requested_level(self, body: dict, default: str = "") -> str:
        """The level a request asks for: reasoning switched off, an effort label the
        profile knows (or one it does not, by its own name), or the default."""
        kwargs = body.get("chat_template_kwargs")
        effort = body.get("reasoning_effort")
        if (isinstance(kwargs, dict) and kwargs.get(self.thinking_kwarg) is False) or effort == "none":
            return self.off_level().name
        if isinstance(effort, str) and effort:
            return next((level.name for level in self.levels if level.effort == effort), effort)
        return default or self.default_level

    def operating_point(self, level: str) -> GovernorConfig | None:
        return self.governor.get(level) or self.governor.get("*")

    def describe(self) -> dict:
        return {
            "id": self.id,
            "levels": self.level_names(),
            "default_level": self.default_level,
            "governed_levels": sorted(self.governor),
            "probe": self.probe is not None,
            "gate_a": self.gate_a,
        }


NONE = EffortLevel("none", thinking=False)

# The operating point measured live on Bonsai 2 at `xhigh` (docs/gate-b-research.md,
# sections 6 to 8). Also used at `medium` until a point measured there replaces it.
MEASURED_XHIGH = GovernorConfig(confidence=0.8, cap=3200, first_probe=300, growth=2.0, anchor=True)

BONSAI_2 = ModelProfile(
    id="bonsai-2",
    match=("*bonsai-2*", "*bonsai2*"),
    levels=(
        NONE,
        EffortLevel("low", effort="low"),
        EffortLevel("medium", effort="medium"),
        EffortLevel("xhigh", effort="xhigh"),
    ),
    # The template's own default. A server started with --reasoning-effort sets
    # another one, which it does not report: FLYBALL_DEFAULT_LEVEL says which.
    default_level="xhigh",
    probe=BOXED,
    governor={"xhigh": MEASURED_XHIGH, "medium": MEASURED_XHIGH},
    notes="Ternary Bonsai 2 27B. Its template rejects any effort it does not know ('high' fails).",
)

# Qwen-style thinking models with a plain on/off switch. Their governor point has
# not been measured: Bonsai 2's applies, as it did before profiles existed.
QWEN_THINKING = ModelProfile(
    id="qwen-thinking",
    match=("*qwen3*", "*qwq*"),
    levels=(NONE, EffortLevel("think")),
    default_level="think",
    probe=BOXED,
    governor={"*": MEASURED_XHIGH},
    notes="Operating point not measured on these models.",
)

# Any model no profile matches: flyball behaves as it did before profiles.
LEGACY = ModelProfile(
    id="legacy",
    match=("*",),
    probe=BOXED,
    governor={"*": MEASURED_XHIGH},
    notes="Unknown model: the defaults measured on Bonsai 2.",
)

BUILT_IN = (BONSAI_2, QWEN_THINKING)


def _level(data: dict) -> EffortLevel:
    return EffortLevel(**{f.name: data[f.name] for f in fields(EffortLevel) if f.name in data})


def profile_from_dict(data: dict, known: Mapping[str, ModelProfile]) -> ModelProfile:
    """A profile from JSON. With `extends`, only the fields given replace the base
    profile's; operating points are merged level by level."""
    base = known.get(data.get("extends", ""))
    if data.get("extends") and base is None:
        raise ValueError("profile {!r} extends unknown profile {!r}".format(data.get("id"), data["extends"]))
    values: dict = {}
    for name in ("id", "default_level", "thinking_kwarg", "template_opens_reasoning", "gate_a", "notes"):
        if name in data:
            values[name] = data[name]
    if "match" in data:
        values["match"] = tuple(data["match"])
    if "reasoning_tags" in data:
        values["reasoning_tags"] = tuple(data["reasoning_tags"])
    if "levels" in data:
        values["levels"] = tuple(_level(level) for level in data["levels"])
    if "probe" in data:
        values["probe"] = None if data["probe"] is None else ProbeProtocol(**data["probe"])
    if "governor" in data:
        points = dict(base.governor) if base is not None else {}
        for level, point in data["governor"].items():
            if point is None:
                points.pop(level, None)
            else:
                points[level] = GovernorConfig.from_dict(point)
        values["governor"] = points
    if base is not None:
        return replace(base, **values)
    if "id" not in values:
        raise ValueError("a profile needs an id")
    return ModelProfile(**values)


def load_profiles(path: Path | str) -> list[ModelProfile]:
    """Profiles from a JSON file: a list, or {"profiles": [...]}. Later entries may
    extend earlier ones and the built-ins."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("profiles", []) if isinstance(raw, dict) else raw
    known: dict[str, ModelProfile] = {p.id: p for p in (*BUILT_IN, LEGACY)}
    out = []
    for entry in entries:
        profile = profile_from_dict(entry, known)
        known[profile.id] = profile
        out.append(profile)
    return out


def all_profiles(extra: list[ModelProfile] | None = None) -> list[ModelProfile]:
    """Loaded profiles first, then the built-ins they do not replace."""
    extra = list(extra or [])
    replaced = {p.id for p in extra}
    return extra + [p for p in BUILT_IN if p.id not in replaced]


def resolve(model_ids: list[str], profiles: list[ModelProfile], forced: str = "") -> ModelProfile:
    """The profile for the first model id any profile matches; LEGACY otherwise."""
    if forced:
        return next((p for p in profiles if p.id == forced), LEGACY)
    for model_id in model_ids:
        if not model_id:
            continue
        for profile in profiles:
            if profile.matches(model_id):
                return profile
    return LEGACY
