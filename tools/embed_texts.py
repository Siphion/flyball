#!/usr/bin/env python3
"""Compute what a candidate System 1 reads in each request, once, for Gate A.

Gate A decides from the request alone. This tool computes, for every problem of a
set of cases, the features a candidate System 1 reads, and the time each takes,
and caches them (.npz) so that gate_a_eval.py can compare candidates without
recomputing anything. The candidates are flyball's Gate A feature specs
(flyball/gate_a.py), so a candidate that wins here runs unchanged in the proxy:

- `state`: flyball's own request features. Free, and the baseline every learned
  reader has to beat;
- `embed:openai:<url>#<model>`: any `/v1/embeddings` endpoint;
- `embed:hf:<name>[@first|@mean]`: a Hugging Face encoder, on the GPU if there is one;
- `embed:laya`: Laya's encoder;
- `systemone:<backend>`: a System One backend asked a few typed questions about
  the request; its probabilities are the features. <backend> is `laya` or a
  server's URL (Laya's server, Jev, ...).

The request is the case's prompt as one user message, read through the same
`request_text` the proxy uses. Features are saved as they are computed, so an
interrupted run resumes. Single-request latency, what a live request would wait,
is measured apart on the first --latency-sample texts, since batching hides it.

Usage:
    python tools/embed_texts.py --cases reports/live_cases.json --cases reports/mmlu_pro_cases_a.json \\
        --features embed:laya --out F:/ai/flyball/research/gate_a_laya.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import evaluation  # noqa: F401 - puts flyball on the path

from flyball.gate_a import extractor
from flyball.state import build_state, request_text


def requests_of(paths: list[Path]) -> tuple[list[str], list[dict]]:
    names, bodies, seen = [], [], set()
    for path in paths:
        for case in json.loads(path.read_text(encoding="utf-8")):
            if case["name"] not in seen:
                seen.add(case["name"])
                names.append(case["name"])
                bodies.append({"messages": [{"role": "user", "content": case["prompt"]}]})
    return names, bodies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--features", required=True, help="a Gate A feature spec, e.g. state or embed:laya")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--latency-sample", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for local encoders")
    args = parser.parse_args()

    try:
        import torch

        torch.set_num_threads(args.threads)  # the CPU overheats when a long job takes every core
    except ImportError:
        pass

    names, bodies = requests_of(args.cases)
    states = [build_state(b) for b in bodies]
    texts = [request_text(b) for b in bodies]
    extract = extractor(args.features)
    partial = args.out.with_suffix(".partial.npz")
    features: list = []
    if partial.exists():
        stored = np.load(partial, allow_pickle=False)
        if list(stored["names"][: len(stored["features"])]) == names[: len(stored["features"])]:
            features = list(stored["features"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("{} requests, {} already done, features {}".format(len(names), len(features), args.features), flush=True)

    began = time.perf_counter()
    for start in range(len(features), len(names), args.batch):
        chunk = slice(start, start + args.batch)
        features.extend(extract(states[chunk], texts[chunk]))
        np.savez_compressed(partial, features=np.array(features, dtype=np.float32), names=np.array(names))
        print("  {}/{} ({:.3f} s per request in batches)".format(
            len(features), len(names), (time.perf_counter() - began) / max(1, len(features) - start)), flush=True)

    latency = []
    for i in range(min(args.latency_sample, len(names))):
        started = time.perf_counter()
        extract(states[i:i + 1], texts[i:i + 1])
        latency.append(time.perf_counter() - started)
    latency = np.array(latency) if latency else np.zeros(1)
    np.savez_compressed(args.out, features=np.array(features, dtype=np.float32), names=np.array(names),
                        latency=latency, spec=np.array(args.features))
    partial.unlink(missing_ok=True)
    print("saved {} x {} features to {}; one request takes {:.0f} ms (median), {:.0f} ms (p95)".format(
        len(features), len(features[0]) if features else 0, args.out,
        1000 * float(np.median(latency)), 1000 * float(np.percentile(latency, 95))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
