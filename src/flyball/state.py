"""StateBuilder: from an OpenAI chat request to a compact, stable description.

Decision backends never see the raw conversation. They see a small set of features
that are cheap to compute, stable across client versions, and safe to log. Keeping
this step pure (no I/O, no clock) makes it testable on its own and keeps the
decision log reproducible: the same request always produces the same state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

# Bounded excerpt of the last user message kept in the state. Long enough to
# classify a request, short enough that a decision log is not a transcript.
EXCERPT_CHARS = 600

# Bump whenever a feature is added or its meaning changes. Every decision row
# carries it, so that a dataset never mixes features computed two different ways.
STATE_VERSION = 2

# A request *for* code, as opposed to code already present in the message.
_CODE_REQUEST = re.compile(
    r"\b(write|implement|create|refactor|generate)\b[^.?!\n]{0,40}"
    r"\b(function|class|method|query|script|regex|program|test|snippet|code)s?\b|"
    r"\b(python|javascript|typescript|sql|rust|golang|java|c\+\+|c#|bash|powershell)\b",
    re.I,
)

_CODE = re.compile(
    r"```|\bdef |\bclass |\bfunction\b|\breturn\b|=>|#include|\bSELECT\b|\bimport\b|;\s*$",
    re.I | re.M,
)
_MATH = re.compile(
    r"\d\s*[-+*/^]\s*\d|\bprobabilit|\bequation|\bsolve\b|\bprove\b|\bmod\b|\bderivative|"
    r"\bintegral|\bhow many\b|\bat what time\b|km/h",
    re.I,
)
_REASONING = re.compile(
    r"\bwhy\b|\bexplain\b|\breason\b|\bdesign\b|\btrade.?off|\bdebug|\bfix\b|\bbug\b|"
    r"\banaly[sz]e|\bcompare\b|\bstep by step\b|\bthink\b",
    re.I,
)
_FORMAT = re.compile(
    r"\bconvert\b|\breformat\b|\brewrite\b|\btranslate\b|\bsort\b|\bto (csv|json|yaml|markdown)\b|"
    r"\bsnake_?case\b|\bcamel_?case\b|\bsummari[sz]e\b|\bnothing else\b|\bonly\b",
    re.I,
)


def _text_of(content) -> tuple[str, int]:
    """Return the text of a message content and how many images it carries."""
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        texts, images = [], 0
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text":
                texts.append(str(part.get("text", "")))
            elif kind in ("image_url", "input_image", "image"):
                images += 1
        return "\n".join(texts), images
    return "", 0


@dataclass(frozen=True)
class RequestState:
    n_messages: int
    n_user_turns: int
    has_system: bool
    has_tools: bool
    n_images: int
    stream: bool
    last_user_chars: int
    total_chars: int
    code_markers: int
    math_markers: int
    reasoning_markers: int
    format_markers: int
    client_effort: str
    client_disabled_thinking: bool
    last_user_hash: str
    last_user_excerpt: str

    def features(self) -> dict:
        """Everything except the free text: what a backend may condition on."""
        out = asdict(self)
        out.pop("last_user_excerpt")
        out.pop("last_user_hash")
        return out


def build_state(body: dict) -> RequestState:
    messages = body.get("messages") or []
    total_chars = 0
    images = 0
    last_user = ""
    user_turns = 0
    has_system = False

    for message in messages:
        if not isinstance(message, dict):
            continue
        text, n_images = _text_of(message.get("content"))
        total_chars += len(text)
        images += n_images
        role = message.get("role")
        if role == "user":
            user_turns += 1
            last_user = text
        elif role in ("system", "developer"):
            has_system = True

    kwargs = body.get("chat_template_kwargs") or {}
    return RequestState(
        n_messages=len(messages),
        n_user_turns=user_turns,
        has_system=has_system,
        has_tools=bool(body.get("tools")),
        n_images=images,
        stream=bool(body.get("stream")),
        last_user_chars=len(last_user),
        total_chars=total_chars,
        code_markers=len(_CODE.findall(last_user)) + len(_CODE_REQUEST.findall(last_user)),
        math_markers=len(_MATH.findall(last_user)),
        reasoning_markers=len(_REASONING.findall(last_user)),
        format_markers=len(_FORMAT.findall(last_user)),
        client_effort=str(body.get("reasoning_effort") or ""),
        client_disabled_thinking=kwargs.get("enable_thinking") is False,
        last_user_hash=hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:16],
        last_user_excerpt=last_user[:EXCERPT_CHARS],
    )
