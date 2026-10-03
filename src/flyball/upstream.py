"""Capability probing: ask the upstream what it can actually do.

flyball must not assume it is talking to llama.cpp, nor that a model honours the
knobs its server advertises. Everything the governor relies on is probed once at
startup and recorded, so an unsupported feature degrades to passthrough instead of
producing a confusing failure halfway through a generation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field

import httpx


@dataclass
class Capabilities:
    """What this upstream supports. Unknown is always treated as unsupported."""

    reachable: bool = False
    auth_required: bool = False
    server: str = ""
    model: str = ""

    # Can the chat template be told to skip thinking? Probed by rendering the same
    # messages twice and comparing: a template that ignores the flag yields the
    # same prompt both times.
    thinking_toggle: bool = False

    # Does the server expose mid-generation reasoning control? This is what the
    # governor needs; without it flyball can only decide before the call.
    reasoning_control: bool = False

    # Static ceiling on thinking tokens, if the server offers one.
    reasoning_budget: bool = False

    # Can the server probe a live completion, asking the model for its answer
    # mid-reasoning? This is what the Gate B governor needs (the `probe` control
    # action; see docs/gate-b-research.md, section 6).
    probe_action: bool = False

    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


async def _json(client: httpx.AsyncClient, method: str, url: str, **kw):
    reply = await client.request(method, url, **kw)
    reply.raise_for_status()
    return reply.json()


async def probe(base_url: str, api_key: str = "", timeout: float = 20.0) -> Capabilities:
    """Interrogate an OpenAI-compatible endpoint. Never raises."""
    caps = Capabilities()
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        try:
            reply = await client.get(f"{base}/health")
            caps.reachable = reply.status_code < 500
        except httpx.RequestError as exc:
            caps.notes.append(f"unreachable: {exc}")
            return caps

        # /props is a llama.cpp extension; absence is informative, not fatal.
        # It doubles as the authentication check: probing further without a key
        # would only collect identical 401s.
        try:
            props = await _json(client, "GET", f"{base}/props")
            caps.model = str(props.get("model_path", "") or props.get("model", ""))
            caps.server = "llama.cpp"
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                caps.auth_required = True
                caps.notes.append(
                    "the upstream requires an API key: set FLYBALL_UPSTREAM_API_KEY "
                    "(capabilities cannot be probed without it)"
                )
                return caps
            caps.notes.append(f"no /props: HTTP {exc.response.status_code}")
        except Exception:
            caps.notes.append("no /props: not a llama.cpp server, or it is disabled")

        await _probe_thinking_toggle(client, base, caps)
        await _probe_reasoning_control(client, base, caps)
        caps.probe_action = await probe_action_supported(client, base)
        if not caps.probe_action:
            caps.notes.append("no probe control action: the governor cannot ask the model mid-reasoning")

    return caps


# A completion id that cannot exist. A server that knows the probe action answers
# that there is no such completion; one that does not know it refuses the action.
# Either way nothing is generated, so the check is cheap enough to repeat.
PROBE_CHECK_ID = "flyball-capability-check"


async def probe_action_supported(
    client: httpx.AsyncClient, base: str, headers: dict | None = None
) -> bool:
    try:
        reply = await client.post(
            f"{base}/v1/chat/completions/control",
            json={"id": PROBE_CHECK_ID, "action": "probe", "suffix": " "},
            headers=headers,
            timeout=5.0,
        )
        if reply.status_code != 200:
            return False
        body = reply.json()
    except Exception:  # noqa: BLE001 - anything unexpected means: not supported
        return False
    return body.get("success") is False and "no active completion" in str(body.get("message", ""))


class ProbeSupport:
    """Whether the upstream offers the probe action, checked lazily and cached.

    Lazily, because the upstream may start after flyball or be swapped for another
    server while flyball runs. A negative answer is kept for less time than a
    positive one, so a server that comes up is picked up quickly.
    """

    def __init__(self, ttl_supported: float = 300.0, ttl_missing: float = 30.0) -> None:
        self.ttl_supported = ttl_supported
        self.ttl_missing = ttl_missing
        self.value = False
        self.checked_at = 0.0

    async def supported(self, client: httpx.AsyncClient, base: str, headers: dict | None = None) -> bool:
        now = time.monotonic()
        ttl = self.ttl_supported if self.value else self.ttl_missing
        if self.checked_at and now - self.checked_at < ttl:
            return self.value
        self.value = await probe_action_supported(client, base, headers)
        self.checked_at = now
        return self.value


async def _probe_thinking_toggle(client: httpx.AsyncClient, base: str, caps: Capabilities) -> None:
    """Render the same messages with thinking on and off and compare the prompts."""
    messages = [{"role": "user", "content": "ping"}]
    rendered = []
    for enabled in (True, False):
        try:
            out = await _json(
                client,
                "POST",
                f"{base}/apply-template",
                json={"messages": messages, "chat_template_kwargs": {"enable_thinking": enabled}},
            )
            rendered.append(out.get("prompt", ""))
        except Exception as exc:
            caps.notes.append(f"no /apply-template, cannot verify the thinking toggle: {exc}")
            return

    if len(rendered) == 2 and rendered[0] and rendered[0] != rendered[1]:
        caps.thinking_toggle = True
    else:
        caps.notes.append("the chat template ignores enable_thinking")


async def _probe_reasoning_control(
    client: httpx.AsyncClient, base: str, caps: Capabilities
) -> None:
    """Arm reasoning control on a throwaway generation and try to end it.

    Costs one very short completion. Worth it: this single capability decides
    whether flyball can govern a generation or merely configure it.
    """
    body = {
        "model": caps.model or "default",
        "stream": True,
        "max_tokens": 64,
        "reasoning_control": True,
        "messages": [{"role": "user", "content": "Think briefly, then say OK."}],
    }
    completion_id = ""
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as reply:
            if reply.status_code >= 400:
                caps.notes.append(f"probe completion refused with HTTP {reply.status_code}")
                return
            async for line in reply.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                import json as _json_mod

                try:
                    event = _json_mod.loads(payload)
                except ValueError:
                    continue
                completion_id = event.get("id") or completion_id
                if completion_id:
                    break
    except Exception as exc:
        caps.notes.append(f"probe completion failed: {exc}")
        return

    if not completion_id:
        caps.notes.append("the stream carried no completion id, reasoning control is unusable")
        return

    try:
        reply = await client.post(
            f"{base}/v1/chat/completions/control",
            json={"id": completion_id, "action": "reasoning_end", "model": body["model"]},
        )
        caps.reasoning_control = reply.status_code == 200 and reply.json().get("success") is True
        if not caps.reasoning_control:
            caps.notes.append(f"control endpoint answered HTTP {reply.status_code}: {reply.text[:120]}")
    except Exception as exc:
        caps.notes.append(f"no control endpoint: {exc}")


def probe_sync(base_url: str, api_key: str = "") -> Capabilities:
    return asyncio.run(probe(base_url, api_key))
