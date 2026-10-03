"""Running chat completions for the research tools, the way the proxy sees them.

`run_completion` streams one completion from any OpenAI-compatible server, reads it
with flyball's own stream reader, and, when given a governor (flyball.governor),
feeds it every chunk exactly as the proxy does. Tools that measure the governor
live therefore measure the code that runs in production, not a copy of it.

The API key is read from the environment or from a file, never from the command
line, where it would end up in shell history and process lists.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flyball.streams import SSEReader  # noqa: E402


def api_key(key_file: str | None = None) -> str:
    if key_file:
        return Path(key_file).read_text(encoding="utf-8").strip()
    return os.environ.get("FLYBALL_UPSTREAM_API_KEY", "")


def client_for(key: str = "") -> httpx.AsyncClient:
    """A client for long local generations: only connecting is bounded."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(10.0, read=None, write=60.0))


async def run_completion(client: httpx.AsyncClient, base: str, body: dict, governor=None,
                         max_seconds: float | None = None) -> dict:
    """Stream one completion to its end, or to `max_seconds`. Keeps the whole text.

    Raises on an HTTP error, so that the caller records the run as failed and can
    retry it, instead of grading an error message as a wrong answer."""
    started = time.perf_counter()
    reader = SSEReader()
    reasoning: list[str] = []
    content: list[str] = []
    ttft = None
    finish = ""
    timed_out = False
    async with client.stream("POST", base.rstrip("/") + "/v1/chat/completions", json=body) as reply:
        if reply.status_code >= 400:
            text = (await reply.aread())[:300]
            raise RuntimeError(f"HTTP {reply.status_code}: {text!r}")
        async for chunk in reply.aiter_raw():
            if governor is not None:
                governor.feed(chunk)
            for delta in reader.feed(chunk):
                reasoning.append(delta.reasoning)
                if (delta.content or delta.tool_calls) and ttft is None:
                    ttft = time.perf_counter() - started
                content.append(delta.content)
                finish = delta.finish_reason or finish
            if max_seconds and time.perf_counter() - started > max_seconds:
                # Leaving the stream closes the connection; the server stops generating.
                timed_out = True
                break
    if governor is not None:
        await governor.close()
    text = "".join(content)
    out = {
        "seconds": round(time.perf_counter() - started, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "reasoning_chars": sum(len(part) for part in reasoning),
        "content_chars": len(text),
        "reasoning": "".join(reasoning),
        "content": text,
        "finish_reason": finish,
        "truncated": timed_out or finish == "length",
        "timed_out": timed_out,
    }
    if governor is not None:
        out["cut"] = governor.cut
        out["probes"] = governor.probes
        out["governor"] = governor.summary()
    return out
