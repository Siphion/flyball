"""Hidden states: what the model itself represents at a point of its reasoning.

The governor asks the model for its answer (governor.py); reading the text of the
reasoning failed (docs/gate-b-research.md, sections 4, 5 and 11). This module is a
third way: take the model's last hidden state at a point of its reasoning, and let
a small linear head say whether the answer has settled. If that works, a governor
needs no probe at all: the state is there at every token, for free.

For now the state is computed offline, again, from the text of the prompt and of
the reasoning so far, by a source that can return it:

- `LlamaCppEmbeddings`: a llama-server started with `--embeddings --pooling last`
  on the same GGUF. That mode switches generation off, so it is a second instance
  for offline work, and it returns the final layer only;
- `TransformersHidden`: any Hugging Face model, any layer, any position, one
  forward pass per text. Needs `flyballd[encoders]`.

vLLM (`extract_hidden_states`) and SGLang (`return_hidden_states`) can be added
behind the same interface. Reading the state live, during generation, needs a
server that streams it; that waits until the offline test says it is worth it.

`LinearProbe` is the trained artifact: a head, its calibration, and a record of
the states it was trained on (source, layer, normalisation, model, level), checked
on load, so that a probe is never applied to states of another kind.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from .calibration import LogisticHead, Platt

PROBE_FORMAT = "flyball.hidden_probe.v1"


class HiddenStateSource(Protocol):
    name: str

    def last_token(self, prompts: list[str]) -> list[list[float]]:
        """The hidden state at the last token of each (already templated) prompt."""
        ...


def decode_base64_floats(data: str) -> list[float]:
    raw = base64.b64decode(data)
    return list(struct.unpack("<{}f".format(len(raw) // 4), raw))


@dataclass
class LlamaCppEmbeddings:
    """A llama-server in embeddings mode, pooling on the last token, unnormalised."""

    base_url: str
    api_key: str = ""
    timeout: float = 600.0
    name: str = "llama.cpp-embeddings"
    layer: str = "final"
    _client: httpx.Client | None = field(default=None, repr=False)

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.Client(timeout=self.timeout, headers=headers)
        return self._client

    def last_token(self, prompts: list[str]) -> list[list[float]]:
        # The OpenAI-style route returns base64 and honours embd_normalize; -1 keeps
        # the state as the model computed it, which is what a linear head was
        # trained on. The prompt's special tokens are parsed, so a rendered chat
        # template comes out as the model saw it.
        reply = self._http().post(self.base_url.rstrip("/") + "/v1/embeddings", json={
            "input": prompts, "encoding_format": "base64", "embd_normalize": -1})
        reply.raise_for_status()
        data = sorted(reply.json()["data"], key=lambda item: item.get("index", 0))
        return [decode_base64_floats(item["embedding"]) for item in data]


@dataclass
class TransformersHidden:
    """Any Hugging Face causal model; `layers` index `hidden_states` (-1 is the last)."""

    model_name: str
    layers: tuple[int, ...] = (-1,)
    device: str = ""
    name: str = "transformers"
    _loaded: tuple | None = field(default=None, repr=False)

    def _load(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
            self._loaded = (torch, tok, model, device)
        return self._loaded

    def _states(self, text: str, token_indices: list[int] | None = None) -> list[list[float]]:
        torch, tok, model, device = self._load()
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            hidden = model(**ids, output_hidden_states=True).hidden_states
        indices = token_indices if token_indices is not None else [ids["input_ids"].shape[1] - 1]
        return [torch.cat([hidden[layer][0, i] for layer in self.layers]).float().cpu().tolist() for i in indices]

    def last_token(self, prompts: list[str]) -> list[list[float]]:
        return [self._states(prompt)[0] for prompt in prompts]

    def at_positions(self, text: str, positions: list[int]) -> list[list[float]]:
        """The states at several character positions of one text, in one pass: a
        trace's checkpoints are prefixes of the same sequence."""
        _, tok, _, _ = self._load()
        offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        indices = []
        for position in positions:
            ended = [i for i, (_, end) in enumerate(offsets) if end <= position]
            indices.append(ended[-1] if ended else 0)
        return self._states(text, indices)


def source_from_spec(spec: str, api_key: str = "") -> HiddenStateSource:
    """`llama.cpp:<base url>` or `hf:<model name>[@<layer>,<layer>...]`."""
    if spec.startswith("llama.cpp:"):
        return LlamaCppEmbeddings(spec[len("llama.cpp:"):], api_key=api_key)
    if spec.startswith("hf:"):
        name, _, layers = spec[len("hf:"):].partition("@")
        return TransformersHidden(name, tuple(int(x) for x in layers.split(",")) if layers else (-1,))
    raise ValueError(f"unknown hidden-state source: {spec!r}")


@dataclass
class LinearProbe:
    """A trained head on hidden states, with what it was trained on."""

    head: LogisticHead
    platt: Platt
    meta: dict  # source, layer, normalisation, model, profile, level, variant

    def predict(self, vector: list[float]) -> float:
        return self.platt.apply(self.head.logit(vector))

    def check(self, expected: dict) -> None:
        """Refuse states of another kind than the probe was trained on."""
        wrong = {k: (self.meta.get(k), v) for k, v in expected.items() if self.meta.get(k) != v}
        if wrong:
            raise ValueError("hidden-state probe trained on other states: " + ", ".join(
                "{} is {!r}, not {!r}".format(k, got, want) for k, (got, want) in wrong.items()))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"format": PROBE_FORMAT, "meta": self.meta, "head": self.head.as_dict(),
                                          "platt": self.platt.as_dict()}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "LinearProbe":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != PROBE_FORMAT:
            raise ValueError(f"{path} is not a hidden-state probe ({PROBE_FORMAT})")
        return cls(LogisticHead.from_dict(data["head"]), Platt.from_dict(data["platt"]), data.get("meta", {}))
