"""One reader for every chat completion flyball watches, whatever the server.

OpenAI-compatible servers agree on the frame (server-sent events: `data: {...}`
lines and a final `data: [DONE]`) but not on where the reasoning goes:

- `reasoning_content`: llama.cpp, SGLang, `transformers serve`;
- `reasoning`: vLLM, the OpenAI routes of Ollama and LM Studio, mlx-lm;
- `thinking`: Ollama's native API;
- nowhere: without a reasoning parser the reasoning stays in the content, between
  tags such as `<think>` and `</think>`.

The reader turns raw bytes into deltas that say which text is reasoning and which
is reply. It never raises on malformed input: a line it cannot read is skipped, so
a quirk of one server costs a statistic, never a response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


@dataclass
class Delta:
    reasoning: str = ""
    content: str = ""
    tool_calls: int = 0
    finish_reason: str = ""


def _reasoning_of(obj: dict) -> str:
    # The first key a server fills is the one it uses; reading all of them would
    # count the same text twice on a server that sends two spellings.
    for key in REASONING_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _count(tool_calls) -> int:
    if isinstance(tool_calls, (list, tuple)):
        return len(tool_calls)
    return 1 if tool_calls else 0


def _partial_tag(text: str, tag: str) -> int:
    """How many characters at the end of `text` could be the start of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class InlineSplitter:
    """Separates reasoning written inline, between tags, from the reply.

    `opened` says whether the reasoning is already open when the stream starts:
    templates that write the opening tag into the prompt (Qwen's, Bonsai's) stream
    only the closing one. A tag split across two chunks is held back until the
    next chunk says what it is.
    """

    def __init__(self, start_tag: str = "<think>", end_tag: str = "</think>", opened: bool = False) -> None:
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.inside = opened
        self._carry = ""

    def split(self, text: str) -> tuple[str, str]:
        """The reasoning and the reply in the next piece of content."""
        text = self._carry + text
        self._carry = ""
        reasoning: list[str] = []
        reply: list[str] = []
        while text:
            tag = self.end_tag if self.inside else self.start_tag
            at = text.find(tag)
            side = reasoning if self.inside else reply
            if at >= 0:
                side.append(text[:at])
                text = text[at + len(tag):]
                self.inside = not self.inside
                continue
            keep = _partial_tag(text, tag)
            side.append(text[: len(text) - keep])
            self._carry = text[len(text) - keep:]
            break
        return "".join(reasoning), "".join(reply)


class SSEReader:
    """Reads a copy of a streamed chat completion, one chunk at a time."""

    def __init__(self, inline: InlineSplitter | None = None) -> None:
        self.inline = inline
        self.completion_id = ""
        self.done = False
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[Delta]:
        self._buffer += chunk
        out: list[Delta] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            out.extend(self._line(line.strip()))
        return out

    def _line(self, line: bytes) -> list[Delta]:
        if not line.startswith(b"data:"):
            return []
        payload = line[5:].strip()
        if payload == b"[DONE]":
            self.done = True
            return []
        try:
            event = json.loads(payload)
        except ValueError:
            return []
        if not isinstance(event, dict):
            return []
        self.completion_id = self.completion_id or str(event.get("id") or "")
        out = []
        for choice in event.get("choices") or []:
            if isinstance(choice, dict):
                out.append(self.delta(choice.get("delta") or {}, choice.get("finish_reason")))
        return out

    def delta(self, fields: dict, finish_reason=None) -> Delta:
        reasoning = _reasoning_of(fields)
        content = fields.get("content")
        content = content if isinstance(content, str) else ""
        if self.inline is not None and content:
            inline_reasoning, content = self.inline.split(content)
            reasoning += inline_reasoning
        return Delta(reasoning, content, _count(fields.get("tool_calls")), finish_reason or "")


def read_message(body: bytes, inline: InlineSplitter | None = None) -> Delta | None:
    """The same reading for a response that was not streamed."""
    try:
        choice = json.loads(body)["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(choice, dict):
        return None
    return SSEReader(inline).delta(choice.get("message") or {}, choice.get("finish_reason"))
