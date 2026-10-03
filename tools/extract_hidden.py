#!/usr/bin/env python3
"""Extract the model's hidden states at every labelled checkpoint of its reasoning.

The probes collected for Gate B (probe_traces.py) say, at each checkpoint of each
reasoning trace, whether the model already had the right answer. This tool
computes the model's own hidden state at those same points, so that
system1_representation.py --features can fit a linear head on it and compare it
with the probe (docs/gate-b-research.md, section 13). Three variants:

- `prefix`: the state at the end of the reasoning so far. What a governor reading
  states live would get for free, at every token, with no probe;
- `suffix`: the state just after the probe's suffix (the reasoning closed, the
  final answer opened). The probe reads its answer from this state, so a head on
  it should come close to the probe: a check of the whole pipeline;
- `prompt`: the state at the end of the prompt, before any reasoning. One per
  problem, for Gate A: does the model itself know a problem is hard?

Prompts are rendered by the generation server's chat template at the effort of
the traces. An embeddings-mode server cannot generate and may not render, so
--render-only renders every prompt first, while the generation server is up, and
--rendered reads them back later.

Resumable: states are saved as they are computed (float16, under the output path,
which belongs on a large disk). With --stop-after-minutes a run stops itself
between two batches.

Usage:
    python tools/extract_hidden.py --render-only --effort xhigh --rendered F:/ai/flyball/research/rendered_xhigh.json \\
        --set math reports/math_traces.jsonl reports/math_cases.json reports/math_probes.jsonl
    python tools/extract_hidden.py --variant prefix --rendered F:/ai/flyball/research/rendered_xhigh.json \\
        --source llama.cpp:http://127.0.0.1:8082 --out F:/ai/flyball/research/hidden_prefix_xhigh.npz \\
        --set math reports/math_traces.jsonl reports/math_cases.json reports/math_probes.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lab import api_key
from system1_representation import load_set

from flyball.governor import BOXED
from flyball.hidden import source_from_spec


def render_all(prompts: list[str], upstream: str, key: str, effort: str, out: Path) -> None:
    import httpx

    stored = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"effort": effort, "prompts": {}}
    if stored["effort"] != effort:
        raise SystemExit(f"{out} holds prompts rendered at effort {stored['effort']}, not {effort}")
    rendered = stored["prompts"]
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(headers=headers, timeout=60.0) as client:
        for prompt in prompts:
            if prompt in rendered:
                continue
            reply = client.post(upstream.rstrip("/") + "/apply-template", json={
                "messages": [{"role": "user", "content": prompt}], "reasoning_effort": effort})
            reply.raise_for_status()
            rendered[prompt] = reply.json()["prompt"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stored), encoding="utf-8")
    print("{} prompts rendered at effort {} into {}".format(len(prompts), effort, out))


def texts_for(rows: list[dict], rendered: dict, variant: str, suffix: str) -> tuple[list[str], list[str]]:
    """The keys and the full texts to read the state of, in a stable order."""
    keys, texts = [], []
    if variant == "prompt":
        for prompt, case in sorted({(r["prompt"], r["case"]) for r in rows}, key=lambda item: item[1]):
            keys.append(case)
            texts.append(rendered[prompt])
        return keys, texts
    for r in rows:
        keys.append("{}|{}|{}".format(r["case"], r["repeat"], r["position"]))
        texts.append(rendered[r["prompt"]] + r["prefix"] + (suffix if variant == "suffix" else ""))
    return keys, texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs=4, action="append", required=True,
                        metavar=("NAME", "TRACES", "CASES", "PROBES"))
    parser.add_argument("--variant", choices=["prefix", "suffix", "prompt"], default="prefix")
    parser.add_argument("--rendered", type=Path, required=True, help="the rendered prompts (JSON)")
    parser.add_argument("--render-only", action="store_true", help="render the prompts and stop")
    parser.add_argument("--effort", default="xhigh", help="the effort the traces were generated at")
    parser.add_argument("--upstream", default="http://127.0.0.1:8081", help="the generation server, for rendering")
    parser.add_argument("--api-key-file", help="read the server key from this file (else FLYBALL_UPSTREAM_API_KEY)")
    parser.add_argument("--source", default="llama.cpp:http://127.0.0.1:8082",
                        help="llama.cpp:<embeddings server> or hf:<model>[@layers]")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="only the first N texts, for a check")
    parser.add_argument("--stop-after-minutes", type=float, default=105.0)
    args = parser.parse_args()

    rows = []
    for name, traces, cases, probes in args.set:
        for row in load_set(Path(traces), Path(cases), Path(probes)):
            row["set"] = name
            rows.append(row)
    key = api_key(args.api_key_file)
    if args.render_only:
        render_all(sorted({r["prompt"] for r in rows}), args.upstream, key, args.effort, args.rendered)
        return 0
    if not args.out:
        parser.error("--out is required to extract")

    stored = json.loads(args.rendered.read_text(encoding="utf-8"))
    if stored["effort"] != args.effort:
        parser.error("the prompts were rendered at effort {}, not {}".format(stored["effort"], args.effort))
    keys, texts = texts_for(rows, stored["prompts"], args.variant, BOXED.suffix)
    if args.limit:
        keys, texts = keys[:args.limit], texts[:args.limit]

    partial = args.out.with_suffix(".partial.npz")
    states: list = []
    if partial.exists():
        stored = np.load(partial, allow_pickle=False)
        if list(stored["keys"]) == keys[: len(stored["states"])]:
            states = list(stored["states"])
    source = source_from_spec(args.source, key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.stop_after_minutes * 60 if args.stop_after_minutes else None
    began, start = time.time(), len(states)
    print("{} texts ({} variant), {} already done, source {}".format(len(keys), args.variant, start, args.source),
          flush=True)
    for i in range(start, len(texts), args.batch):
        if deadline and time.time() > deadline:
            print("stopping: --stop-after-minutes reached; run again to resume", flush=True)
            return 0
        states.extend(np.asarray(v, dtype=np.float16) for v in source.last_token(texts[i:i + args.batch]))
        if len(states) % (args.batch * 25) < args.batch or len(states) == len(texts):
            np.savez(partial, states=np.stack(states), keys=np.array(keys[: len(states)]))
            print("  {}/{} ({:.2f} s each)".format(len(states), len(texts), (time.time() - began) / (len(states) - start)),
                  flush=True)
    np.savez_compressed(args.out, states=np.stack(states), keys=np.array(keys), variant=np.array(args.variant),
                        source=np.array(args.source), effort=np.array(args.effort))
    partial.unlink(missing_ok=True)
    print("saved {} states of width {} to {}".format(len(states), len(states[0]), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
