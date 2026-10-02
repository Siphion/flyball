#!/usr/bin/env python3
"""The Gate B teacher: would cutting the reasoning here still give the right answer?

For each uncut reasoning trace, the reasoning is truncated at regular checkpoints,
the thinking block is closed by hand, and the model is asked to answer from that
point, greedily. If the forced answer is correct, cutting there was safe.

That is the exact question the governor has to answer, measured rather than
guessed: no regular expression over the reasoning, no judge model. It is also
expensive, because every label costs a short generation on the GPU, which is why it
is the teacher and not the governor. The governor is a small model on the CPU that
learns to predict these labels from the partial reasoning alone.

Two details matter for the labels to mean anything:

- the prompt is rendered with the same reasoning effort used to produce the trace,
  because the chat template changes with it;
- checkpoints are moved back to the previous whitespace, so that the model is never
  asked to continue from half a word.

Resumable: rows already present in the output file are skipped.

Usage:
    python tools/probe_traces.py --traces reports/gsm8k_traces.jsonl \\
        --cases reports/gsm8k_cases.json --out reports/gsm8k_probes.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from measure_reasoning_cut import last_number


class Upstream:
    def __init__(self, base_url: str, api_key: str, effort: str) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.effort = effort
        self._rendered: dict[str, str] = {}

    def _post(self, path: str, body: dict, timeout: float = 600.0) -> dict:
        request = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), self.headers, method="POST"
        )
        return json.load(urllib.request.urlopen(request, timeout=timeout))

    def rendered(self, prompt: str) -> str:
        if prompt not in self._rendered:
            self._rendered[prompt] = self._post(
                "/apply-template",
                {"messages": [{"role": "user", "content": prompt}], "reasoning_effort": self.effort},
            )["prompt"]
        return self._rendered[prompt]

    def forced_answer(self, prompt: str, reasoning_prefix: str, max_tokens: int) -> str:
        body = {
            "prompt": self.rendered(prompt) + reasoning_prefix + "\n</think>\n\n",
            "n_predict": max_tokens,
            "temperature": 0,
            # Successive probes on one trace share a growing prefix: the server only
            # has to process the characters added since the previous checkpoint.
            "cache_prompt": True,
        }
        return self._post("/completion", body)["content"]


def checkpoints(reasoning: str, step: int, max_probes: int) -> list[int]:
    """Positions to probe, every `step` characters, snapped back to whitespace."""
    positions = []
    for target in range(step, len(reasoning), step):
        cut = reasoning.rfind(" ", 0, target)
        cut = max(cut, reasoning.rfind("\n", 0, target))
        if cut > 0 and (not positions or cut > positions[-1]):
            positions.append(cut)
    if len(positions) > max_probes:
        # Evenly thinned, always keeping the earliest and the latest.
        stride = (len(positions) - 1) / (max_probes - 1)
        positions = sorted({positions[round(i * stride)] for i in range(max_probes)})
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key", default=os.environ.get("FLYBALL_UPSTREAM_API_KEY", ""))
    parser.add_argument("--effort", default="xhigh", help="must match the effort of the traces")
    parser.add_argument("--step", type=int, default=200, help="characters between checkpoints")
    parser.add_argument("--max-probes", type=int, default=12, help="per trace")
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    args = parser.parse_args()

    cases = {c["name"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}
    traces = [
        json.loads(line)
        for line in args.traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [t for t in traces if not t.get("cut") and t["case"] in cases]

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["case"], row["repeat"], row["position"]))

    upstream = Upstream(args.upstream, args.api_key, args.effort)
    total = sum(len(checkpoints(t["reasoning"], args.step, args.max_probes)) for t in traces)
    print("{} traces, {} checkpoints, {} already labelled".format(len(traces), total, len(done)))

    count = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for trace in traces:
            case = cases[trace["case"]]
            reference = last_number(case["reference_answer"])
            for position in checkpoints(trace["reasoning"], args.step, args.max_probes):
                count += 1
                key = (trace["case"], trace["repeat"], position)
                if key in done:
                    continue
                started = time.time()
                try:
                    answer = upstream.forced_answer(
                        case["prompt"], trace["reasoning"][:position], args.max_answer_tokens
                    )
                except Exception as exc:  # noqa: BLE001 - one failed probe is one missing label
                    print("  [{}/{}] {} @{} FAILED: {!r}".format(count, total, trace["case"], position, exc))
                    continue
                safe = last_number(answer) == reference
                handle.write(json.dumps({
                    "case": trace["case"],
                    "repeat": trace["repeat"],
                    "position": position,
                    "reasoning_chars": len(trace["reasoning"]),
                    "fraction": round(position / len(trace["reasoning"]), 4),
                    "safe": safe,
                    "trace_correct": trace["correct"],
                    "probe_seconds": round(time.time() - started, 2),
                    "answer_tail": answer[-200:],
                }, ensure_ascii=False) + "\n")
                handle.flush()
                if count % 25 == 0:
                    print("  [{}/{}] {} @{} safe={}".format(count, total, trace["case"], position, safe))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
