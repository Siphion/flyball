"""Gate B, live: close the reasoning once the model has settled on its answer.

The governor reads a copy of a streamed chat completion, never the stream itself:
the client receives the upstream's bytes unchanged. While the reasoning runs, it
asks the model what its final answer would be, through the server's `probe`
control action, once the reasoning passes 300 characters and each time it doubles.
As soon as the model is confident enough in an answer that is a number, or once the
reasoning reaches a cap, the governor closes the reasoning with `reasoning_end`,
and the model writes its reply.

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
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

log = logging.getLogger("flyball.governor")

# Closes the reasoning and starts the reply with a final-answer box, stopping just
# before its opening brace: tokenisers merge the brace with what follows it ("{-" is
# one token for Qwen), so ending the prompt on "{" would keep the model from writing
# a minus sign.
PROBE_SUFFIX = "\n</think>\n\n**Final Answer**\n\n\\boxed"
PROBE_TOKENS = 12

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LATEX_SEPARATOR = re.compile(r"(?<=\d)(?:\{,\}|\\[,!;: ])(?=\d)")

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


@dataclass
class GovernorConfig:
    confidence: float = 0.8
    cap: int = 3200
    first_probe: int = 300
    # Close the reasoning with a sentence that commits to the answer, so that the
    # reply does not pick the reasoning back up. Needs `reasoning_end` to accept a
    # message; a server that does not simply ignores it.
    anchor: bool = True


def anchor_message(reason: str, answer: str | None) -> str:
    """The last sentence of the reasoning, forced in just before it closes."""
    if reason == "confident" and answer is not None:
        return "\n\nSo the final answer is {}.\n".format(answer)
    return "\n\nI'll stop here and give my best final answer.\n"


@dataclass
class Governor:
    """Watches one streamed completion and decides when to close its reasoning."""

    config: GovernorConfig
    control: Control
    model: str = ""
    completion_id: str = ""
    reasoning_chars: int = 0
    replying: bool = False
    cut: dict | None = None
    probes: list = field(default_factory=list)
    stopped: str = ""
    _next_probe: int = 0
    _pending: asyncio.Task | None = None
    _buffer: bytes = b""
    _started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._next_probe = self.config.first_probe

    # -- reading the stream -------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        """Called with every chunk the client receives. Never raises."""
        if self.stopped:
            return
        try:
            self._buffer += chunk
            while b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                self._line(line.strip())
            self._act()
        except Exception as exc:  # noqa: BLE001 - the stream must never pay for a governor bug
            log.exception("governor stopped after an error")
            self.stopped = "error: {!r}".format(exc)

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except ValueError:
            return
        self.completion_id = self.completion_id or str(event.get("id") or "")
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                self.reasoning_chars += len(delta["reasoning_content"])
            if delta.get("content") or delta.get("tool_calls"):
                self.replying = True

    # -- deciding -------------------------------------------------------------

    def _act(self) -> None:
        if self.cut or self.replying or not self.completion_id:
            return
        if self._pending is not None and not self._pending.done():
            return  # one control call at a time
        if self.reasoning_chars >= self.config.cap:
            self._spawn(self._end("cap"))
        elif self.reasoning_chars >= self._next_probe:
            while self._next_probe <= self.reasoning_chars:
                self._next_probe *= 2
            self._spawn(self._probe())

    def _spawn(self, coroutine) -> None:
        self._pending = asyncio.get_running_loop().create_task(coroutine)

    async def _probe(self) -> None:
        at = self.reasoning_chars
        started = time.time()
        reply = await self._call({
            "id": self.completion_id,
            "model": self.model,
            "action": "probe",
            "suffix": PROBE_SUFFIX,
            "n_predict": PROBE_TOKENS,
            "stop": "}",
        })
        record = {"at": at, "ms": round((time.time() - started) * 1000, 1), "ok": bool(reply.get("success"))}
        self.probes.append(record)
        if not record["ok"]:
            record["message"] = str(reply.get("message", ""))[:200]
            if self.replying:
                return  # the reasoning ended on its own while the probe was in flight
            self.stopped = "probe refused: " + record["message"]
            return
        probe = reply.get("probe") or {}
        record["answer"] = answer_value(str(probe.get("content", "")))
        record["p"] = round(math.exp(float(probe.get("logprob", -math.inf))), 4)
        if record["answer"] is not None and record["p"] >= self.config.confidence:
            await self._end("confident", record["answer"])

    async def _end(self, reason: str, answer: str | None = None) -> None:
        if self.cut or self.replying:
            return
        at = self.reasoning_chars
        payload = {"id": self.completion_id, "action": "reasoning_end", "model": self.model}
        if self.config.anchor:
            payload["message"] = anchor_message(reason, answer)
        reply = await self._call(payload)
        self.cut = {"reason": reason, "at": at, "ok": bool(reply.get("success")),
                    "after_s": round(time.time() - self._started, 3)}
        if self.config.anchor:
            self.cut["message"] = payload["message"].strip()
        if not self.cut["ok"]:
            self.stopped = "reasoning_end refused: " + str(reply.get("message", ""))[:200]

    async def _call(self, payload: dict) -> dict:
        try:
            return await self.control(payload)
        except Exception as exc:  # noqa: BLE001 - a failed call is a reason to stop, not to crash
            return {"success": False, "message": repr(exc)}

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
            "reasoning_chars": self.reasoning_chars,
            "probes": self.probes,
            "cut": self.cut,
            "stopped": self.stopped,
        }
