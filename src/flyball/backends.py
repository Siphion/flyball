"""Decision backends: the System 1 behind flyball's gates, and swappable by design.

flyball does not depend on one decision model. Every backend speaks the same
contract, the System One protocol: a state (text), a set of typed questions
(`choice` among options, `score` on a scale, `noul` for yes or no), and answers
that carry probabilities. TypeSafe's hosted Jev defines it on the wire
(`POST /v1/systemone`), and Laya serves the same format, so one HTTP backend talks
to either, or to anything else that serves it.

Two backends:

- `SystemOneHTTP`: any server that speaks the protocol. Laya's own server
  (`laya-serve`), Jev's hosted API, or another implementation, by URL.
- `LayaLocal`: Laya's Python SDK in this process, on the CPU or a GPU. Optional:
  it imports Laya only when used.

`backend_from_spec("laya")` or `backend_from_spec("https://host/...")` picks one.

The answers come back as the protocol returns them, one dict per state, keyed by
question name, for example `{"settled": {"noul": 0.83}}`. Calibrating them, and
deciding what a probability is worth, is flyball's job, not the backend's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class Backend(Protocol):
    name: str

    def decide(self, states: list[str], questions: dict) -> list[dict]:
        """One answers dict per state, in the same order."""
        ...


def _state(text: str) -> dict:
    return {"body": text}


@dataclass
class SystemOneHTTP:
    """A server that speaks the System One protocol, such as laya-serve or Jev."""

    base_url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 60.0
    batch_size: int = 64  # laya-serve caps a batch at 64 states
    name: str = "systemone-http"
    _client: httpx.Client | None = field(default=None, repr=False)

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.Client(timeout=self.timeout, headers=headers)
        return self._client

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def decide(self, states: list[str], questions: dict) -> list[dict]:
        out: list[dict] = []
        for start in range(0, len(states), self.batch_size):
            chunk = states[start:start + self.batch_size]
            body: dict = {"questions": questions}
            if self.model:
                body["model"] = self.model
            if len(chunk) == 1:
                body["state"] = _state(chunk[0])
                reply = self._http().post(self._url("/v1/systemone"), json=body)
                reply.raise_for_status()
                out.append(reply.json()["answers"])
            else:
                body["states"] = [_state(s) for s in chunk]
                reply = self._http().post(self._url("/v1/systemone/batch"), json=body)
                reply.raise_for_status()
                out.extend(result["answers"] for result in reply.json()["results"])
        return out


@dataclass
class LayaLocal:
    """Laya's SDK in this process. `model` names a checkpoint (english,
    multilingual, typed-decisions) or leaves the choice to Laya's router."""

    model: str = ""
    threads: int = 0
    max_len: int = 0
    batch_size: int = 16
    name: str = "laya-local"
    _router: object = field(default=None, repr=False)

    def _load(self):
        if self._router is None:
            if self.threads:
                import torch

                torch.set_num_threads(self.threads)
            from laya import Router  # optional dependency: pip install laya

            self._router = Router()
        return self._router

    def decide(self, states: list[str], questions: dict) -> list[dict]:
        router = self._load()
        options: dict = {}
        if self.model:
            options["model"] = self.model
        if self.max_len:
            options["max_len"] = self.max_len
        # Laya's router batches requests, each carrying its own state and questions,
        # and routes them per checkpoint; the results come back in input order.
        requests = [{"state": state, "questions": questions, **options} for state in states]
        results = router.predict_batch(requests, batch_size=self.batch_size, sort_by_length=True)
        return [result["answers"] for result in results]


def backend_from_spec(spec: str) -> Backend:
    """`laya`, `laya:<checkpoint>`, or the base URL of a System One server.
    An HTTP server's key is read from FLYBALL_SYSTEMONE_API_KEY."""
    if spec == "laya" or spec.startswith("laya:"):
        return LayaLocal(model=spec.partition(":")[2])
    if spec.startswith(("http://", "https://")):
        return SystemOneHTTP(spec, api_key=os.environ.get("FLYBALL_SYSTEMONE_API_KEY", ""))
    raise ValueError(f"unknown decision backend: {spec!r}")


def noul(answers: dict, question: str) -> float:
    """The probability of yes, from an answers dict."""
    return float(answers[question]["noul"])
