"""Hidden states: read the same way from any source, and a probe never applied to
states of another kind."""

from __future__ import annotations

import base64
import json
import struct

import httpx
import pytest

from flyball.calibration import LogisticHead, Platt
from flyball.hidden import LinearProbe, LlamaCppEmbeddings, decode_base64_floats, source_from_spec


def encoded(values: list[float]) -> str:
    return base64.b64encode(struct.pack("<{}f".format(len(values)), *values)).decode()


def test_llama_cpp_states_are_asked_for_unnormalised_and_decoded() -> None:
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        data = [{"index": i, "embedding": encoded([float(i), 0.5, -2.0])} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    source = LlamaCppEmbeddings("http://embeddings.invalid")
    source._client = httpx.Client(transport=httpx.MockTransport(handle))
    states = source.last_token(["<|im_start|>user\nhi", "second"])
    assert states == [[0.0, 0.5, -2.0], [1.0, 0.5, -2.0]]
    assert seen[0] == {"input": ["<|im_start|>user\nhi", "second"], "encoding_format": "base64", "embd_normalize": -1}


def test_base64_floats_round_trip() -> None:
    assert decode_base64_floats(encoded([1.5, -3.25])) == [1.5, -3.25]


def test_sources_are_picked_by_spec() -> None:
    assert isinstance(source_from_spec("llama.cpp:http://127.0.0.1:8082"), LlamaCppEmbeddings)
    assert source_from_spec("hf:some/model@-1,-8").layers == (-1, -8)
    with pytest.raises(ValueError):
        source_from_spec("vllm:http://x")


def test_a_probe_round_trips_and_refuses_other_states(tmp_path) -> None:
    probe = LinearProbe(LogisticHead([0.0, 0.0], [1.0, 1.0], [2.0, 0.0], 0.0), Platt(),
                        {"source": "llama.cpp:http://x", "variant": "prefix", "effort": "xhigh"})
    path = tmp_path / "probe.json"
    probe.save(path)
    loaded = LinearProbe.load(path)
    assert loaded.predict([1.0, 0.0]) == pytest.approx(probe.predict([1.0, 0.0]))
    loaded.check({"variant": "prefix", "effort": "xhigh"})
    with pytest.raises(ValueError, match="effort"):
        loaded.check({"effort": "medium"})


def test_transformers_reads_several_positions_in_one_pass() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from flyball.hidden import TransformersHidden

    source = TransformersHidden("sshleifer/tiny-gpt2", device="cpu")
    try:
        states = source.at_positions("one two three four", [3, 7, 18])
    except OSError:
        pytest.skip("the tiny test model is not available offline")
    assert len(states) == 3 and len(states[0]) == len(states[2])
