"""Decision backends: the System 1 behind flyball's gates, and swappable by design.

flyball does not depend on one decision model. Two kinds of backend exist, and
any implementation of either can be swapped for another.

**Backends that answer questions** speak the System One protocol: a state (text,
or a structured object), a set of typed questions (`choice` among options, `score`
on a scale, `noul` for yes or no), and answers that carry probabilities.
TypeSafe's hosted Jev defines it on the wire (`POST /v1/systemone`). Laya's server
serves the same format, and so do ollaya, llama.cpp and Ollama with a decision
model, and SGLang with any chat model. One HTTP backend therefore talks to all of
them. There are three implementations:

- `SystemOneHTTP`: any server that speaks the protocol, by URL;
- `LayaLocal`: Laya's Python SDK in this process (optional, imported when used);
- `HeuristicBackend`: flyball's own thirty-line heuristic, behind the same
  interface, as the baseline every other backend has to beat.

**Embedders** turn text into a vector, on which flyball fits its own small head
(calibration.py). There are three implementations:

- `OpenAIEmbeddings`: the `/v1/embeddings` route of any server (llama.cpp, Ollama,
  vLLM, LM Studio, a dedicated embedding server);
- `HFEncoder`: any Hugging Face encoder (optional);
- `LayaEncoder`: Laya's encoder (optional).

Answers come back as the protocol returns them. Calibrating them, and deciding
what a probability is worth, is flyball's job, not the backend's. In the request
path every call goes through `ask`, which turns a slow or failing backend into an
abstention, never into a delay or an error.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeVar

import httpx

T = TypeVar("T")


class Backend(Protocol):
    name: str

    def decide(self, states: list, questions: dict) -> list[dict]:
        """One answers dict per state, in the same order."""
        ...


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _state(state) -> dict:
    return state if isinstance(state, dict) else {"body": state}


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

    def decide(self, states: list, questions: dict) -> list[dict]:
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

    def decide(self, states: list, questions: dict) -> list[dict]:
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


@dataclass
class HeuristicBackend:
    """flyball's heuristic (heuristic.py), answering the protocol's questions it
    knows. A state is a chat request body, or plain text taken as one user turn."""

    name: str = "heuristic"

    def decide(self, states: list, questions: dict) -> list[dict]:
        from .heuristic import SCHEMA, decide
        from .state import build_state

        out = []
        for state in states:
            body = state if isinstance(state, dict) and "messages" in state else {
                "messages": [{"role": "user", "content": state.get("body", "") if isinstance(state, dict) else state}]}
            decision = decide(build_state(body))
            answers: dict = {}
            for name in questions:
                if name == "needs_deliberation":
                    answers[name] = {"type": "noul", "noul": decision.probabilities[name]["true"]}
                elif name == "request_class":
                    dist = decision.probabilities[name]
                    answers[name] = {"type": "choice", "choice": decision.values[name], "probabilities": dist}
                elif name == "effort":
                    levels = SCHEMA["effort"]["levels"]
                    chosen = decision.values[name]
                    answers[name] = {"type": "score", "score": levels.index(chosen),
                                     "probabilities": {level: float(level == chosen) for level in levels}}
            out.append(answers)
        return out


def backend_from_spec(spec: str) -> Backend:
    """`heuristic`, `laya`, `laya:<checkpoint>`, or the base URL of a System One
    server. An HTTP server's key is read from FLYBALL_SYSTEMONE_API_KEY."""
    if spec == "heuristic":
        return HeuristicBackend()
    if spec == "laya" or spec.startswith("laya:"):
        return LayaLocal(model=spec.partition(":")[2])
    if spec.startswith(("http://", "https://")):
        return SystemOneHTTP(spec, api_key=os.environ.get("FLYBALL_SYSTEMONE_API_KEY", ""))
    raise ValueError(f"unknown decision backend: {spec!r}")


def noul(answers: dict, question: str) -> float:
    """The probability of yes, from an answers dict."""
    return float(answers[question]["noul"])


def distribution(answer: dict, levels: list[str] | None = None) -> dict[str, float]:
    """An answer of any type as a probability per option.

    A noul answer gives `true` and `false`. A choice gives its probabilities. A
    score gives its probabilities keyed by level name: servers key them by level
    name or by position ("0", "1", ...), and `levels` maps positions to names. The
    backend's own `confidence` is ignored on purpose; flyball calibrates instead.
    """
    if "noul" in answer:
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    probabilities = answer.get("probabilities") or {}
    out: dict[str, float] = {}
    for key, value in probabilities.items():
        name = str(key)
        if levels is not None and name.isdigit() and int(name) < len(levels):
            name = levels[int(name)]
        out[name] = float(value)
    if not out and "choice" in answer:
        out[str(answer["choice"])] = 1.0
    if not out and "score" in answer and levels is not None:
        score = answer["score"]
        if isinstance(score, int) and 0 <= score < len(levels):
            out[levels[score]] = 1.0
    return out


# -- embedders ----------------------------------------------------------------


@dataclass
class OpenAIEmbeddings:
    """`POST /v1/embeddings` on any server that offers it."""

    base_url: str
    model: str = ""
    api_key: str = ""
    timeout: float = 60.0
    batch_size: int = 32
    name: str = "openai-embeddings"
    _client: httpx.Client | None = field(default=None, repr=False)

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.Client(timeout=self.timeout, headers=headers)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            body: dict = {"input": texts[start:start + self.batch_size]}
            if self.model:
                body["model"] = self.model
            reply = self._http().post(self.base_url.rstrip("/") + "/v1/embeddings", json=body)
            reply.raise_for_status()
            data = sorted(reply.json()["data"], key=lambda item: item.get("index", 0))
            out.extend([float(v) for v in item["embedding"]] for item in data)
        return out


@dataclass
class HFEncoder:
    """A Hugging Face encoder, mean- or first-token-pooled. Needs `flyballd[encoders]`."""

    model_name: str
    pooling: str = "mean"
    max_len: int = 512
    device: str = ""
    batch_size: int = 16
    name: str = "hf-encoder"
    _loaded: tuple | None = field(default=None, repr=False)

    def _load(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name).eval().to(device)
            self._loaded = (torch, tok, model, device)
        return self._loaded

    def embed(self, texts: list[str]) -> list[list[float]]:
        torch, tok, model, device = self._load()
        out: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = tok(texts[start:start + self.batch_size], return_tensors="pt", truncation=True,
                            max_length=self.max_len, padding=True).to(device)
                hidden = model(**batch).last_hidden_state
                if self.pooling == "first":
                    pooled = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                out.extend(pooled.float().cpu().tolist())
        return out


@dataclass
class LayaEncoder:
    """Laya's encoder (ModernBERT): the first token's state and the mean over all
    tokens, as system1_representation.py reads them. Needs `laya`."""

    threads: int = 0
    max_len: int = 512
    batch_size: int = 16
    name: str = "laya-encoder"
    _loaded: tuple | None = field(default=None, repr=False)

    def _load(self):
        if self._loaded is None:
            import torch
            import laya

            if self.threads:
                torch.set_num_threads(self.threads)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = laya.load("convaiinnovations/laya")
            encoder = agent.model.encoder.eval().to(device)
            if device == "cuda":
                encoder = encoder.half()
            self._loaded = (torch, agent.tok, encoder, device)
        return self._loaded

    def embed(self, texts: list[str]) -> list[list[float]]:
        torch, tok, encoder, device = self._load()
        out: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = tok(texts[start:start + self.batch_size], return_tensors="pt", truncation=True,
                            max_length=self.max_len, padding=True).to(device)
                hidden = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                out.extend(torch.cat([hidden[:, 0], mean], dim=1).float().cpu().tolist())
        return out


def embedder_from_spec(spec: str) -> Embedder:
    """`openai:<base url>#<model>`, `hf:<model name>[@first|@mean]`, or `laya`.
    An embeddings server's key is read from FLYBALL_EMBEDDINGS_API_KEY."""
    if spec == "laya":
        return LayaEncoder()
    if spec.startswith("openai:"):
        url, _, model = spec[len("openai:"):].partition("#")
        return OpenAIEmbeddings(url, model=model, api_key=os.environ.get("FLYBALL_EMBEDDINGS_API_KEY", ""))
    if spec.startswith("hf:"):
        name, _, pooling = spec[len("hf:"):].partition("@")
        return HFEncoder(name, pooling=pooling or "mean")
    raise ValueError(f"unknown embedder: {spec!r}")


# -- in the request path --------------------------------------------------------


async def ask(fn: Callable[..., T], *args, timeout: float) -> T | None:
    """Run a blocking backend call off the event loop, within `timeout` seconds.

    None means abstain: the call was too slow or failed. The request never waits
    longer than the timeout and never sees the error. A call that overruns keeps
    running in its thread until it finishes, and its answer is dropped.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
    except Exception:  # noqa: BLE001 - timeouts and backend errors alike mean: no decision
        return None
