"""Gate B, live: close the reasoning once the model has settled on its answer.

The governor reads a copy of a streamed chat completion, never the stream itself:
the client receives the upstream's bytes unchanged. While the reasoning runs, it
asks the model what its final answer would be, through the server's controller,
once the reasoning passes a first threshold (300 characters by default) and each
time it grows by a factor (doubling by default). As soon as the model is confident
enough in an answer it can vouch for (a number, an option letter), or once the
reasoning reaches a cap, the governor closes the reasoning, and the model writes
its reply.

What the governor needs from the outside comes through two small interfaces, so
that it is tied to no server and no model:

- a `Controller` (servers.py) ends the reasoning and probes the live completion.
  llama.cpp's control endpoint is one implementation; a server without mid-stream
  control answers `unsupported`, and the governor simply stops;
- a `ProbeProtocol` (one per model family, in profiles.py) says how to ask a model
  for its answer: what closes its reasoning, what opens the answer, where to stop,
  and how to read what comes back.

Why this design, and not a classifier reading the reasoning: measured in
docs/gate-b-research.md. A probe costs about 0.2 seconds of generation; asked
sparingly it saves more than it costs, the model's confidence is well calibrated,
and live it beat a fixed reasoning budget on time at no cost in accuracy.

Fail-open throughout: a probe that errors, a control call that is refused, any
exception at all, and the governor simply stops governing. The generation carries
on as if it had never been there.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, runtime_checkable

from .streams import SSEReader

log = logging.getLogger("flyball.governor")

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LATEX_SEPARATOR = re.compile(r"(?<=\d)(?:\{,\}|\\[,!;: ])(?=\d)")

# The original interface: an async callable that takes a control payload for
# llama.cpp's `/v1/chat/completions/control` and returns its JSON reply.
Control = Callable[[dict], Awaitable[dict]]


def answer_number(text: str) -> str | None:
    """The number in a probe's answer, or None if there is none to trust."""
    found = _NUMBER.findall(_LATEX_SEPARATOR.sub("", text))
    if not found:
        return None
    value = found[-1].replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


# An option letter as a model boxes it: {B, {\text{B, {(B), {B. ...
_OPTION = re.compile(r"^\W*(?:\\text\w*\{)?\W*([A-J])\b")


def answer_value(text: str) -> str | None:
    """A probe's answer when it is short enough to trust: a number, or the letter of
    a multiple-choice option. Anything longer is not an answer the probe can vouch for."""
    number = answer_number(text)
    if number is not None:
        return number
    found = _OPTION.match(text)
    return found.group(1) if found else None


PARSERS: dict[str, Callable[[str], str | None]] = {
    "number_or_option": answer_value,
    "number": answer_number,
}


@dataclass(frozen=True)
class ProbeProtocol:
    """How to ask one family of models for its answer, mid-reasoning.

    `suffix` closes the reasoning and opens the answer. The default closes a
    `<think>` block and opens a final-answer box, stopping just before its opening
    brace: tokenisers merge the brace with what follows it ("{-" is one token for
    Qwen), so ending the prompt on "{" would keep the model from writing a minus
    sign. The anchors are the last sentence forced into the reasoning when the
    governor closes it, so that the reply does not pick the reasoning back up.
    """

    suffix: str = "\n</think>\n\n**Final Answer**\n\n\\boxed"
    stop: str = "}"
    n_predict: int = 12
    parser: str = "number_or_option"
    anchor_confident: str = "\n\nSo the final answer is {answer}.\n"
    anchor_cap: str = "\n\nI'll stop here and give my best final answer.\n"

    def parse(self, text: str) -> str | None:
        return PARSERS.get(self.parser, answer_value)(text)

    def as_dict(self) -> dict:
        return {"suffix": self.suffix, "stop": self.stop, "n_predict": self.n_predict, "parser": self.parser,
                "anchor_confident": self.anchor_confident, "anchor_cap": self.anchor_cap}


# Models that think in a <think> block and box their final answer (Qwen and its
# derivatives, Bonsai 2 among them). Measured in docs/gate-b-research.md.
BOXED = ProbeProtocol()

# The names the first version exported, kept for the tools that import them.
PROBE_SUFFIX = BOXED.suffix
PROBE_TOKENS = BOXED.n_predict


@dataclass(frozen=True)
class GovernorConfig:
    """An operating point: when to probe, how sure is sure enough, where to stop.

    Probes start once the reasoning reaches `first_probe` characters and repeat each
    time it grows by `growth`; `first_probe=None` means no probes at all, which
    leaves a plain budget at `cap`. `cap=None` means no cap.
    """

    confidence: float = 0.8
    cap: int | None = 3200
    first_probe: int | None = 300
    growth: float = 2.0
    # Close the reasoning with a sentence that commits to the answer, so that the
    # reply does not pick the reasoning back up. Needs the end of the reasoning to
    # accept a message; a server that does not simply ignores it.
    anchor: bool = True

    def as_dict(self) -> dict:
        return {"confidence": self.confidence, "cap": self.cap, "first_probe": self.first_probe,
                "growth": self.growth, "anchor": self.anchor}

    @classmethod
    def from_dict(cls, data: dict) -> "GovernorConfig":
        known = {k: data[k] for k in ("confidence", "cap", "first_probe", "growth", "anchor") if k in data}
        return cls(**known)


def next_probe(position: int, growth: float) -> int:
    """The probe point after `position`: always strictly further along."""
    return max(position + 1, int(round(position * growth)))


def probe_points(config: GovernorConfig, limit: int) -> list[int]:
    """Every point at which the governor would probe a reasoning of `limit`
    characters. Shared with the offline simulation, so that both use one schedule."""
    if config.first_probe is None:
        return []
    points, at = [], config.first_probe
    # At the cap the governor ends the reasoning instead of probing it.
    while at <= limit and (config.cap is None or at < config.cap):
        points.append(at)
        at = next_probe(at, config.growth)
    return points


@dataclass
class ControlResult:
    """What a controller reports back. `unsupported` means the server cannot do
    this at all, which is not an error: the governor stops quietly."""

    ok: bool
    unsupported: bool = False
    message: str = ""
    content: str = ""  # a probe's answer
    logprob: float = -math.inf  # its log-probability


@runtime_checkable
class Controller(Protocol):
    async def end_reasoning(self, completion_id: str, message: str | None) -> ControlResult: ...

    async def probe(self, completion_id: str, protocol: ProbeProtocol) -> ControlResult: ...


def anchor_message(reason: str, answer: str | None, protocol: ProbeProtocol = BOXED) -> str:
    """The last sentence of the reasoning, forced in just before it closes."""
    if reason == "confident" and answer is not None:
        return protocol.anchor_confident.format(answer=answer)
    return protocol.anchor_cap


@dataclass
class Governor:
    """Watches one streamed completion and decides when to close its reasoning."""

    config: GovernorConfig
    # A Controller, or a bare `Control` callable, taken to be llama.cpp's control
    # endpoint (the interface of the first version).
    control: Controller | Control
    model: str = ""
    protocol: ProbeProtocol | None = BOXED
    completion_id: str = ""
    reasoning_chars: int = 0
    replying: bool = False
    cut: dict | None = None
    probes: list = field(default_factory=list)
    stopped: str = ""
    _next_probe: int | None = None
    _pending: asyncio.Task | None = None
    _reader: SSEReader = field(default_factory=SSEReader)
    _started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._next_probe = self.config.first_probe if self.protocol is not None else None
        if not isinstance(self.control, Controller):
            from .servers import LlamaCppController  # imported here: servers.py imports this module

            self.control = LlamaCppController(post=self.control, model=self.model)

    # -- reading the stream -------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        """Called with every chunk the client receives. Never raises."""
        if self.stopped:
            return
        try:
            for delta in self._reader.feed(chunk):
                self.reasoning_chars += len(delta.reasoning)
                if delta.content or delta.tool_calls:
                    self.replying = True
            self.completion_id = self.completion_id or self._reader.completion_id
            self._act()
        except Exception as exc:  # noqa: BLE001 - the stream must never pay for a governor bug
            log.exception("governor stopped after an error")
            self.stopped = "error: {!r}".format(exc)

    # -- deciding -------------------------------------------------------------

    def _act(self) -> None:
        if self.cut or self.replying or not self.completion_id:
            return
        if self._pending is not None and not self._pending.done():
            return  # one control call at a time
        if self.config.cap is not None and self.reasoning_chars >= self.config.cap:
            self._spawn(self._end("cap"))
        elif self._next_probe is not None and self.reasoning_chars >= self._next_probe:
            while self._next_probe <= self.reasoning_chars:
                self._next_probe = next_probe(self._next_probe, self.config.growth)
            self._spawn(self._probe())

    def _spawn(self, coroutine) -> None:
        self._pending = asyncio.get_running_loop().create_task(coroutine)

    async def _probe(self) -> None:
        at = self.reasoning_chars
        started = time.time()
        result = await self._guard(self.control.probe(self.completion_id, self.protocol))
        record = {"at": at, "ms": round((time.time() - started) * 1000, 1), "ok": result.ok}
        self.probes.append(record)
        if not result.ok:
            record["message"] = result.message[:200]
            if self.replying:
                return  # the reasoning ended on its own while the probe was in flight
            self.stopped = ("probe unsupported: " if result.unsupported else "probe refused: ") + record["message"]
            return
        record["answer"] = self.protocol.parse(result.content)
        record["p"] = round(math.exp(result.logprob), 4)
        if record["answer"] is not None and record["p"] >= self.config.confidence:
            await self._end("confident", record["answer"])

    async def _end(self, reason: str, answer: str | None = None) -> None:
        if self.cut or self.replying:
            return
        at = self.reasoning_chars
        message = anchor_message(reason, answer, self.protocol or BOXED) if self.config.anchor else None
        result = await self._guard(self.control.end_reasoning(self.completion_id, message))
        self.cut = {"reason": reason, "at": at, "ok": result.ok,
                    "after_s": round(time.time() - self._started, 3)}
        if message is not None:
            self.cut["message"] = message.strip()
        if not result.ok:
            self.stopped = "reasoning_end refused: " + result.message[:200]

    @staticmethod
    async def _guard(call) -> ControlResult:
        try:
            return await call
        except Exception as exc:  # noqa: BLE001 - a failed call is a reason to stop, not to crash
            return ControlResult(False, message=repr(exc))

    # -- closing ----------------------------------------------------------------

    async def close(self) -> None:
        """Wait briefly for a control call still in flight, then let it go."""
        if self._pending is not None and not self._pending.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._pending), timeout=1.0)
            except Exception:  # noqa: BLE001 - nothing left to govern anyway
                self._pending.cancel()

    def summary(self) -> dict:
        return {
            "confidence": self.config.confidence,
            "cap": self.config.cap,
            "first_probe": self.config.first_probe,
            "growth": self.config.growth,
            "reasoning_chars": self.reasoning_chars,
            "probes": self.probes,
            "cut": self.cut,
            "stopped": self.stopped,
        }
