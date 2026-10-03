"""Server adapters: how flyball talks to the server that runs the model.

flyball is not tied to one inference server. Everything it needs from one goes
through an adapter:

- **discovery**: what the server is, which model it runs, what it can do;
- **rendering**: turning an abstract reasoning level (profiles.py) into the fields
  of a request, since every server spells "think less" its own way;
- **arming**: whatever a request needs so that it can be governed later;
- **control**: ending the reasoning of a live completion and probing it, through a
  `Controller`.

Mid-stream control comes in four kinds, and "none" is a first-class answer, not an
error:

- `in_place`: the server ends or probes the live completion itself. llama.cpp's
  `POST /v1/chat/completions/control` does (the probe action needs flyball's patch);
- `reissue`: abort, then continue the reasoning as a prefix in a new request.
  Cheap on servers that cache prefixes (vLLM, SGLang, mlx-lm); not built yet;
- `side_request`: probe through a separate request carrying the prefix; not built;
- `none`: the governor stays off, and only Gate A can act.

Two adapters exist. `LlamaCpp` does everything. `OpenAICompatible` serves any
other OpenAI-compatible server: it renders levels into the fields most of them
accept (`chat_template_kwargs`, `reasoning_effort`) and reports that it cannot
control a live completion.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Protocol

import httpx

from .governor import ControlResult, Controller, ProbeProtocol
from .profiles import EffortLevel, ModelProfile

IN_PLACE = "in_place"
REISSUE = "reissue"
SIDE_REQUEST = "side_request"
NONE = "none"

KINDS = ("auto", "llama.cpp", "openai")

Post = Callable[[dict], Awaitable[dict]]


class Unrenderable(ValueError):
    """A level this server cannot express."""


@dataclass
class ServerCapabilities:
    """What one upstream supports. Unknown is always treated as unsupported."""

    kind: str = ""
    model_id: str = ""
    model_alias: str = ""
    reachable: bool = False
    auth_required: bool = False
    thinking_toggle: bool | None = None  # None: not checked
    effort_field: bool | None = None  # the template understands reasoning_effort
    budget_field: str = ""  # the request field for a reasoning budget; "" = none
    end_reasoning: str = NONE
    end_message: bool = False  # ending the reasoning can force a closing sentence in
    probe: str = NONE
    hidden_states: str = NONE
    notes: list[str] = field(default_factory=list)

    def model_names(self) -> list[str]:
        return [name for name in (self.model_alias, self.model_id) if name]

    def as_dict(self) -> dict:
        return asdict(self)


# -- controllers --------------------------------------------------------------


@dataclass
class LlamaCppController:
    """Ends and probes a live completion through llama.cpp's control endpoint."""

    post: Post
    model: str = ""

    async def _call(self, payload: dict) -> dict:
        try:
            return await self.post(payload)
        except Exception as exc:  # noqa: BLE001 - a failed call is an answer, not a crash
            return {"success": False, "message": repr(exc)}

    async def probe(self, completion_id: str, protocol: ProbeProtocol) -> ControlResult:
        reply = await self._call({
            "id": completion_id,
            "model": self.model,
            "action": "probe",
            "suffix": protocol.suffix,
            "n_predict": protocol.n_predict,
            "stop": protocol.stop,
        })
        if not reply.get("success"):
            return ControlResult(False, message=str(reply.get("message", "")))
        probe = reply.get("probe") or {}
        return ControlResult(True, content=str(probe.get("content", "")),
                             logprob=float(probe.get("logprob", -math.inf)))

    async def end_reasoning(self, completion_id: str, message: str | None) -> ControlResult:
        payload = {"id": completion_id, "action": "reasoning_end", "model": self.model}
        if message is not None:
            payload["message"] = message
        reply = await self._call(payload)
        return ControlResult(bool(reply.get("success")), message=str(reply.get("message", "")))


@dataclass
class UnsupportedController:
    """For a server that cannot touch a live completion: says so, without I/O."""

    why: str = "this server has no mid-stream control"

    async def probe(self, completion_id: str, protocol: ProbeProtocol) -> ControlResult:
        return ControlResult(False, unsupported=True, message=self.why)

    async def end_reasoning(self, completion_id: str, message: str | None) -> ControlResult:
        return ControlResult(False, unsupported=True, message=self.why)


# -- adapters -----------------------------------------------------------------


class ServerAdapter(Protocol):
    kind: str

    async def discover(self, client: httpx.AsyncClient, base: str, headers: dict,
                       check_probe: bool = False) -> ServerCapabilities: ...

    def render_level(self, body: dict, level: EffortLevel, profile: ModelProfile) -> dict: ...

    def arm(self, body: dict) -> dict: ...

    def controller(self, client: httpx.AsyncClient, base: str, headers: dict, model: str) -> Controller: ...


async def _get(client: httpx.AsyncClient, url: str, headers: dict, caps: ServerCapabilities):
    """GET a discovery endpoint. Returns the reply, or None when the server cannot
    be reached; records an authentication failure on the way."""
    try:
        reply = await client.get(url, headers=headers, timeout=5.0)
    except httpx.RequestError as exc:
        caps.notes.append(f"unreachable: {exc}")
        return None
    caps.reachable = reply.status_code < 500
    if reply.status_code in (401, 403):
        caps.auth_required = True
        caps.notes.append("the upstream requires an API key: set FLYBALL_UPSTREAM_API_KEY")
    return reply


class OpenAICompatible:
    """Any OpenAI-compatible server: levels rendered into the common fields, no
    control of a live completion."""

    kind = "openai"
    budget_field = ""

    def render_level(self, body: dict, level: EffortLevel, profile: ModelProfile) -> dict:
        """The request asking for `level`. Pure; raises Unrenderable when the level
        needs something this server has no field for."""
        out = dict(body)
        kwargs = body.get("chat_template_kwargs")
        if not level.thinking:
            out["chat_template_kwargs"] = dict(kwargs or {}, **{profile.thinking_kwarg: False})
            if out.get("reasoning_effort"):
                del out["reasoning_effort"]
        else:
            if isinstance(kwargs, dict) and kwargs.get(profile.thinking_kwarg) is False:
                out["chat_template_kwargs"] = dict(kwargs, **{profile.thinking_kwarg: True})
            if level.effort:
                out["reasoning_effort"] = level.effort
        if level.budget_tokens:
            if not self.budget_field:
                raise Unrenderable(f"{self.kind} has no per-request reasoning budget")
            out[self.budget_field] = level.budget_tokens
        return out

    def arm(self, body: dict) -> dict:
        return body

    def controller(self, client: httpx.AsyncClient, base: str, headers: dict, model: str) -> Controller:
        return UnsupportedController()

    async def discover(self, client: httpx.AsyncClient, base: str, headers: dict,
                       check_probe: bool = False) -> ServerCapabilities:
        caps = ServerCapabilities(kind=self.kind)
        reply = await _get(client, f"{base}/v1/models", headers, caps)
        if reply is not None and reply.status_code == 200:
            try:
                models = reply.json().get("data") or []
                caps.model_id = str(models[0].get("id", "")) if models else ""
            except (ValueError, AttributeError, TypeError):
                pass
        return caps


# A completion id that cannot exist. A server that knows the probe action answers
# that there is no such completion; one that does not know it refuses the action.
# Either way nothing is generated, so the check is cheap enough to repeat.
PROBE_CHECK_ID = "flyball-capability-check"


async def probe_action_supported(client: httpx.AsyncClient, base: str, headers: dict | None = None) -> bool:
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


class LlamaCpp(OpenAICompatible):
    """llama.cpp's server: `reasoning_control` arms a completion, the control
    endpoint ends it, and flyball's patch adds the probe action
    (docs/gate-b-research.md, section 6)."""

    kind = "llama.cpp"
    # Per request, overriding --reasoning-budget. In the server's source, not in
    # its documentation, so G0 checks it before anything relies on it.
    budget_field = "reasoning_budget_tokens"

    def arm(self, body: dict) -> dict:
        out = dict(body)
        out["reasoning_control"] = True
        return out

    def controller(self, client: httpx.AsyncClient, base: str, headers: dict, model: str) -> Controller:
        async def post(payload: dict) -> dict:
            reply = await client.post(f"{base}/v1/chat/completions/control", json=payload,
                                      headers=headers, timeout=10.0)
            try:
                data = reply.json()
            except ValueError:
                data = {}
            if reply.status_code != 200:
                return {"success": False, "message": f"HTTP {reply.status_code}: {str(data)[:120]}"}
            return data

        return LlamaCppController(post=post, model=model)

    async def discover(self, client: httpx.AsyncClient, base: str, headers: dict,
                       check_probe: bool = False) -> ServerCapabilities:
        caps = ServerCapabilities()
        reply = await _get(client, f"{base}/props", headers, caps)
        if reply is None or caps.auth_required:
            return caps
        if reply.status_code == 200:
            try:
                props = reply.json()
            except ValueError:
                props = None
            if isinstance(props, dict):
                caps.kind = self.kind
                caps.model_id = str(props.get("model_path") or props.get("model") or "")
                caps.model_alias = str(props.get("model_alias") or "")
                template = props.get("chat_template_caps") or {}
                if isinstance(template, dict) and "supports_reasoning_effort" in template:
                    caps.effort_field = bool(template["supports_reasoning_effort"])
                caps.budget_field = self.budget_field
                # Stock since the control endpoint was merged; `flyballd probe`
                # verifies it with a throwaway generation.
                caps.end_reasoning = IN_PLACE
        else:
            caps.notes.append(f"no /props: HTTP {reply.status_code}")
        if check_probe:
            if await probe_action_supported(client, base, headers):
                caps.kind = self.kind
                caps.probe = IN_PLACE
                caps.end_reasoning = IN_PLACE
                caps.end_message = True  # the patch that adds the probe also adds the message
            else:
                caps.notes.append("no probe control action: the governor cannot ask the model mid-reasoning")
        return caps


def adapter_for(kind: str) -> OpenAICompatible:
    if kind == "llama.cpp":
        return LlamaCpp()
    if kind == "openai":
        return OpenAICompatible()
    raise ValueError(f"unknown server kind {kind!r}; expected one of {KINDS}")


class Discovery:
    """What the upstream is and what it can do, discovered lazily and cached.

    Lazily, because the upstream may start after flyball or be swapped for another
    server while flyball runs. An answer that leaves something missing is kept for
    less time than a complete one, so a server that comes up is picked up quickly.
    """

    def __init__(self, kind: str = "auto", ttl_ok: float = 300.0, ttl_missing: float = 30.0) -> None:
        if kind not in KINDS:
            raise ValueError(f"FLYBALL_SERVER must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.ttl_ok = ttl_ok
        self.ttl_missing = ttl_missing
        self.adapter: OpenAICompatible = LlamaCpp() if kind != "openai" else OpenAICompatible()
        self.caps = ServerCapabilities()
        self.checked_at = 0.0
        self._complete = False

    async def get(self, client: httpx.AsyncClient, base: str, headers: dict,
                  check_probe: bool = False) -> tuple[OpenAICompatible, ServerCapabilities]:
        now = time.monotonic()
        ttl = self.ttl_ok if self._complete else self.ttl_missing
        if self.checked_at and now - self.checked_at < ttl:
            return self.adapter, self.caps
        if self.kind == "openai":
            adapter: OpenAICompatible = OpenAICompatible()
            caps = await adapter.discover(client, base, headers)
        else:
            adapter = LlamaCpp()
            caps = await adapter.discover(client, base, headers, check_probe)
            if self.kind == "auto" and caps.kind != adapter.kind and not caps.auth_required:
                fallback = OpenAICompatible()
                generic = await fallback.discover(client, base, headers)
                generic.notes = caps.notes + generic.notes
                adapter, caps = fallback, generic
        self.adapter, self.caps, self.checked_at = adapter, caps, now
        self._complete = caps.reachable and (caps.probe == IN_PLACE or not check_probe)
        return adapter, caps


# -- the full check, for `flyballd probe` --------------------------------------


async def _thinking_toggle(client: httpx.AsyncClient, base: str, caps: ServerCapabilities) -> None:
    """Render the same messages with thinking on and off and compare the prompts."""
    messages = [{"role": "user", "content": "ping"}]
    rendered = []
    for enabled in (True, False):
        try:
            reply = await client.post(f"{base}/apply-template",
                                      json={"messages": messages, "chat_template_kwargs": {"enable_thinking": enabled}})
            reply.raise_for_status()
            rendered.append(reply.json().get("prompt", ""))
        except Exception as exc:  # noqa: BLE001
            caps.notes.append(f"no /apply-template, cannot verify the thinking toggle: {exc}")
            return
    caps.thinking_toggle = bool(rendered[0]) and rendered[0] != rendered[1]
    if not caps.thinking_toggle:
        caps.notes.append("the chat template ignores enable_thinking")


async def _reasoning_end_works(client: httpx.AsyncClient, base: str, caps: ServerCapabilities) -> None:
    """Arm reasoning control on a throwaway generation and try to end it.

    Costs one very short completion. Worth it: this single capability decides
    whether flyball can govern a generation or merely configure it.
    """
    body = {
        "model": caps.model_alias or caps.model_id or "default",
        "stream": True,
        "max_tokens": 64,
        "reasoning_control": True,
        "messages": [{"role": "user", "content": "Think briefly, then say OK."}],
    }
    completion_id = ""
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as reply:
            if reply.status_code >= 400:
                caps.notes.append(f"test completion refused with HTTP {reply.status_code}")
                caps.end_reasoning = NONE
                return
            async for line in reply.aiter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                try:
                    completion_id = json.loads(line[6:]).get("id") or completion_id
                except ValueError:
                    continue
                if completion_id:
                    break
    except Exception as exc:  # noqa: BLE001
        caps.notes.append(f"test completion failed: {exc}")
        caps.end_reasoning = NONE
        return
    if not completion_id:
        caps.notes.append("the stream carried no completion id, reasoning control is unusable")
        caps.end_reasoning = NONE
        return
    try:
        reply = await client.post(f"{base}/v1/chat/completions/control",
                                  json={"id": completion_id, "action": "reasoning_end", "model": body["model"]})
        works = reply.status_code == 200 and reply.json().get("success") is True
        caps.end_reasoning = IN_PLACE if works else NONE
        if not works:
            caps.notes.append(f"control endpoint answered HTTP {reply.status_code}: {reply.text[:120]}")
    except Exception as exc:  # noqa: BLE001
        caps.end_reasoning = NONE
        caps.notes.append(f"no control endpoint: {exc}")


async def probe(base_url: str, api_key: str = "", kind: str = "auto", timeout: float = 20.0) -> ServerCapabilities:
    """Everything flyball can learn about an upstream, including the checks that
    cost a short generation. Never raises."""
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        _, caps = await Discovery(kind).get(client, base, {}, check_probe=True)
        if caps.kind == "llama.cpp" and not caps.auth_required:
            await _thinking_toggle(client, base, caps)
            await _reasoning_end_works(client, base, caps)
    return caps


def probe_sync(base_url: str, api_key: str = "", kind: str = "auto") -> ServerCapabilities:
    return asyncio.run(probe(base_url, api_key, kind))
