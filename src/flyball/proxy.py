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

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings

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


async def _forward(app: FastAPI, settings: Settings, path: str, request: Request) -> Response:
    client: httpx.AsyncClient = app.state.client
    url = settings.upstream_url(path)
    headers = _forwardable(request.headers, drop_auth=bool(settings.upstream_api_key))
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    body = await request.body()
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

    async def passthrough() -> AsyncIterator[bytes]:
        try:
            async for chunk in reply.aiter_raw():
                yield chunk
        finally:
            await reply.aclose()

    return StreamingResponse(
        passthrough(),
        status_code=reply.status_code,
        headers=_forwardable(reply.headers, drop_auth=False),
        media_type=reply.headers.get("content-type"),
    )
