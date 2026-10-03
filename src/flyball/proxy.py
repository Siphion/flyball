"""An OpenAI-compatible reverse proxy that stays out of the way.

Two rules govern this module:

1. **Fail open.** flyball sits in the critical path of whatever talks to your model
   (a chat UI, a coding agent, a script). Anything flyball cannot do, it must not
   break: on any internal error the request is forwarded unchanged.
2. **Byte-exact passthrough.** Request and response bodies are streamed through
   without being parsed or re-serialised, so tool calls, multimodal content,
   logprobs and vendor-specific fields survive untouched.

In `off` mode this module is the whole program.
"""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .governor import Governor
from .ledger import Ledger
from .profiles import all_profiles, load_profiles
from .servers import IN_PLACE, Discovery
from .shadow import OutcomeTracker, observe

log = logging.getLogger("flyball.proxy")

# Headers that describe a single hop and must not be copied to the other side.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


# Reachable without a key, as on llama-server: a liveness probe has to work before
# any client has been configured.
PUBLIC_PATHS = frozenset({"health", "v1/health"})


def _client_key(headers) -> str:
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return headers.get("x-api-key", "").strip()


def _authorised(settings: Settings, path: str, request: Request) -> bool:
    """When FLYBALL_API_KEY is set, every client must present it.

    This is what keeps an exposed flyball from becoming an open door to the model:
    the upstream's own key is replaced on the way through, so the upstream can no
    longer tell an authorised client from anyone else on the network.
    """
    if not settings.api_key:
        return True
    # CORS preflights never carry credentials; refusing them breaks browsers
    # without protecting anything.
    if request.method == "OPTIONS" or path.strip("/") in PUBLIC_PATHS:
        return True
    supplied = _client_key(request.headers)
    return bool(supplied) and hmac.compare_digest(supplied.encode(), settings.api_key.encode())


def _forwardable(headers: httpx.Headers | dict, drop_auth: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        low = key.lower()
        if low in HOP_BY_HOP:
            continue
        if drop_auth and low == "authorization":
            continue
        out[key] = value
    return out


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            settings.connect_timeout,
            read=settings.read_timeout,
            write=settings.connect_timeout,
            pool=settings.connect_timeout,
        )
        app.state.client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(
        title="flyball",
        version="0.0.1",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    # In off mode nothing is recorded: no ledger, no files, no observation.
    app.state.ledger = Ledger(settings.data_dir) if settings.mode != "off" else None
    app.state.discovery = Discovery(settings.server)
    app.state.profiles = all_profiles(load_profiles(settings.profiles_path) if settings.profiles_path else [])
    if settings.profile and settings.profile != "legacy" and settings.profile not in {
            p.id for p in app.state.profiles}:
        raise ValueError(f"FLYBALL_PROFILE names an unknown profile: {settings.profile!r}")

    @app.get("/flyball/health")
    async def health() -> JSONResponse:
        """flyball's own health, plus whether the upstream answers."""
        upstream_ok = False
        try:
            reply = await app.state.client.get(settings.upstream_url("/health"), timeout=3.0)
            upstream_ok = reply.status_code < 500
        except Exception:  # upstream being down is a reportable state, not an error
            upstream_ok = False
        return JSONResponse(
            {
                "status": "ok",
                "mode": settings.mode,
                "kill_switch": settings.kill_switch.exists(),
                "thresholds": settings.thresholds,
                "governor": {
                    "enabled": settings.mode == "active" and settings.governor,
                    "probe_supported": app.state.discovery.caps.probe == IN_PLACE,
                    "overrides": settings.governor_overrides(),
                },
                "server": {
                    "kind": app.state.discovery.caps.kind,
                    "model": app.state.discovery.caps.model_alias or app.state.discovery.caps.model_id,
                    "default_level": settings.default_level,
                },
                "profiles": [p.describe() for p in app.state.profiles],
                "upstream": settings.upstream,
                "upstream_reachable": upstream_ok,
            }
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def forward(path: str, request: Request) -> Response:
        return await _forward(app, settings, path, request)

    return app


def _start_governor(
    client: httpx.AsyncClient, settings: Settings, body: bytes, headers: dict, tracker: OutcomeTracker
) -> Governor | None:
    """A governor for one completion, at the operating point the model's profile
    gives for the request's reasoning level, adjusted by any override that is set."""
    try:
        model = str(json.loads(body).get("model") or "")
    except (ValueError, AttributeError):
        model = ""
    point = tracker.profile.operating_point(tracker.level)
    if point is None or tracker.adapter is None:
        return None
    overrides = settings.governor_overrides()
    if overrides:
        point = replace(point, **overrides)
    controller = tracker.adapter.controller(client, settings.upstream, headers, model)
    return Governor(config=point, control=controller, model=model, protocol=tracker.profile.probe)


async def _forward(app: FastAPI, settings: Settings, path: str, request: Request) -> Response:
    if not _authorised(settings, path, request):
        # Same shape llama-server uses, so clients show a familiar error.
        return JSONResponse(
            {
                "error": {
                    "code": 401,
                    "message": "Invalid API Key",
                    "type": "authentication_error",
                }
            },
            status_code=401,
        )

    client: httpx.AsyncClient = app.state.client
    url = settings.upstream_url(path)
    headers = _forwardable(request.headers, drop_auth=bool(settings.upstream_api_key))
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    body = await request.body()
    control_headers = (
        {"Authorization": f"Bearer {settings.upstream_api_key}"} if settings.upstream_api_key else {}
    )

    # The decision is taken before the request leaves, for two reasons: a
    # non-streaming upstream only answers once generation is over, so timing has
    # to start here; and in active mode the decision will reshape the request.
    tracker = None
    ledger = getattr(app.state, "ledger", None)
    if (
        ledger is not None
        and request.method == "POST"
        and path.rstrip("/").endswith("chat/completions")
    ):
        try:
            probe_supported = False
            adapter = caps = None
            if settings.mode == "active" and settings.governor:
                adapter, caps = await app.state.discovery.get(
                    client, settings.upstream, control_headers, check_probe=True
                )
                probe_supported = caps.probe == IN_PLACE
            tracker, body = observe(ledger, settings, body, probe_supported=probe_supported,
                                    adapter=adapter, caps=caps, profiles=app.state.profiles)
        except Exception:  # observation must never cost a response
            # `body` is only reassigned on success, so a failure here always
            # forwards the original request.
            log.exception("observation failed; forwarding untouched")
            tracker = None

    upstream_request = client.build_request(
        request.method,
        url,
        params=request.query_params,
        headers=headers,
        content=body,
    )

    try:
        reply = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        log.warning("upstream unreachable: %s", exc)
        return JSONResponse(
            {
                "error": {
                    "message": f"flyball could not reach the upstream at {settings.upstream}",
                    "type": "upstream_unavailable",
                }
            },
            status_code=502,
        )

    governor = None
    if tracker is not None:
        tracker.status = reply.status_code
        if "content-encoding" in reply.headers:
            # A compressed body cannot be read from a raw copy; the outcome row
            # will carry the status and timings only.
            tracker.unreadable = True
        plan = tracker.governor_plan
        if plan is not None and plan.govern and reply.status_code == 200 and not tracker.unreadable:
            governor = _start_governor(client, settings, body, control_headers, tracker)

    async def passthrough() -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in reply.aiter_raw():
                if tracker is not None:
                    try:
                        tracker.feed(chunk)
                    except Exception:
                        log.exception("outcome tracking failed for one chunk")
                if governor is not None:
                    # Reads a copy and never raises; the chunk goes out unchanged.
                    governor.feed(chunk)
                yield chunk
            completed = True
        finally:
            await reply.aclose()
            if governor is not None:
                await governor.close()
            if tracker is not None:
                try:
                    outcome = tracker.finish(completed)
                    if governor is not None:
                        outcome["governor"] = governor.summary()
                    ledger.outcomes.write(outcome)
                except Exception:
                    log.exception("could not record the outcome")

    return StreamingResponse(
        passthrough(),
        status_code=reply.status_code,
        headers=_forwardable(reply.headers, drop_auth=False),
        media_type=reply.headers.get("content-type"),
    )
