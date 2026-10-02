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

from . import heuristic
from .ledger import Ledger
from .state import build_state


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
    _buffer: bytes = b""
    _body: list = field(default_factory=list)

    def feed(self, chunk: bytes) -> None:
        if self.unreadable:
            return
        if not self.streaming:
            self._body.append(chunk)
            return
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._line(line.strip())

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if payload == b"[DONE]":
            self.saw_done = True
            return
        try:
            event = json.loads(payload)
        except ValueError:
            return
        for choice in event.get("choices") or []:
            self._delta(choice.get("delta") or {})
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def _delta(self, delta: dict) -> None:
        if delta.get("reasoning_content"):
            self.reasoning_chars += len(delta["reasoning_content"])
        if delta.get("content"):
            if self.first_content_at is None:
                self.first_content_at = time.time()
            self.content_chars += len(delta["content"])
        if delta.get("tool_calls"):
            if self.first_content_at is None:
                self.first_content_at = time.time()
            self.tool_calls += len(delta["tool_calls"])

    def finish(self, completed: bool) -> dict:
        if not self.streaming and self._body:
            try:
                message = json.loads(b"".join(self._body))["choices"][0]
                self._delta(message.get("message") or {})
                self.finish_reason = message.get("finish_reason") or ""
                self.saw_done = True
            except (ValueError, KeyError, IndexError, TypeError):
                pass
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


def observe(ledger: Ledger, mode: str, raw_body: bytes) -> OutcomeTracker | None:
    """Record a decision for a chat completion request. Returns a tracker, or None
    when the request is not something flyball reasons about."""
    try:
        body = json.loads(raw_body)
    except ValueError:
        return None
    if not isinstance(body, dict) or not body.get("messages"):
        return None

    state = build_state(body)
    started = time.time()
    decision = heuristic.decide(state)
    decided_in = time.time() - started

    decision_id = uuid.uuid4().hex
    ledger.decisions.write(
        {
            "decision_id": decision_id,
            "ts": round(started, 3),
            "mode": mode,
            "applied": False,
            "model": body.get("model", ""),
            "schema_sha256": heuristic.SCHEMA_SHA256,
            "backend": decision.backend,
            "decide_ms": round(decided_in * 1000, 3),
            "values": decision.values,
            "probabilities": decision.probabilities,
            "confidence": decision.confidence,
            "features": state.features(),
            "last_user_hash": state.last_user_hash,
            "last_user_excerpt": state.last_user_excerpt,
        }
    )
    return OutcomeTracker(decision_id=decision_id, started=started, streaming=state.stream)
