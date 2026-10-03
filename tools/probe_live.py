#!/usr/bin/env python3
"""Exercise the server-side probe on live completions: what it costs, and whether it
leaves the generation untouched.

The probe is a control action, next to `reasoning_end`, on a llama.cpp server built
with it (see docs/gate-b-research.md, section 5): while a completion is streaming, it
appends a suffix after everything generated so far, decodes a few tokens greedily
from the model's own distribution, returns them with their log-probabilities, and
restores the slot as it was.

Two things are measured, per problem:

- **invisibility**: the same problem is generated greedily (temperature 0) without
  probes and with a probe every few hundred characters of reasoning. The two texts
  must be identical, byte for byte. Any difference means the probe leaked into the
  generation;
- **cost**: each probe's latency as the client sees it, the server's own breakdown
  (saving the state, decoding, restoring it), and the difference in total time.

Usage:
    python tools/probe_live.py --cases reports/math_cases.json \\
        --names math-train-number_theory-790 math-test-algebra-2556
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path

SUFFIX = "\n</think>\n\n**Final Answer**\n\n\\boxed"


class Server:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key

    def post(self, path: str, body: dict, timeout: float = 600.0):
        request = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), self.headers, method="POST"
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def probe(self, completion_id: str, n_predict: int) -> dict:
        started = time.perf_counter()
        with self.post("/v1/chat/completions/control", {
            "id": completion_id,
            "action": "probe",
            "suffix": SUFFIX,
            "n_predict": n_predict,
            "stop": "}",
        }, timeout=60) as response:
            reply = json.load(response)
        reply["client_ms"] = (time.perf_counter() - started) * 1000
        return reply


def generate(server: Server, prompt: str, effort: str, max_tokens: int,
             probe_every: int, n_predict: int) -> dict:
    """Stream one greedy completion, probing every `probe_every` characters of reasoning."""
    body = {
        "model": "default",
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
        "messages": [{"role": "user", "content": prompt}],
    }
    reasoning, content = [], []
    probes, threads = [], []
    completion_id = None
    next_probe = probe_every
    started = time.perf_counter()

    def run_probe(at: int) -> None:
        try:
            reply = server.probe(completion_id, n_predict)
        except Exception as exc:  # noqa: BLE001 - a failed probe is recorded, not fatal
            reply = {"success": False, "message": repr(exc)}
        reply["at_chars"] = at
        probes.append(reply)

    with server.post("/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            completion_id = completion_id or chunk.get("id")
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
            so_far = sum(len(part) for part in reasoning)
            if probe_every and not content and so_far >= next_probe:
                # In a thread: the stream must keep being read while the probe waits.
                thread = threading.Thread(target=run_probe, args=(so_far,))
                thread.start()
                threads.append(thread)
                next_probe = so_far + probe_every
    seconds = time.perf_counter() - started
    for thread in threads:
        thread.join()
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "seconds": seconds,
        "probes": sorted(probes, key=lambda p: p["at_chars"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key", default=os.environ.get("FLYBALL_UPSTREAM_API_KEY", ""))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--probe-every", type=int, default=300, help="characters of reasoning")
    parser.add_argument("--n-predict", type=int, default=12)
    parser.add_argument("--out", type=Path, help="write every run as JSON lines")
    args = parser.parse_args()

    cases = {c["name"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}
    server = Server(args.upstream, args.api_key)
    out = args.out.open("w", encoding="utf-8") if args.out else None

    client_ms, server_ms, identical = [], [], []
    for name in args.names:
        case = cases[name]
        plain = generate(server, case["prompt"], args.effort, args.max_tokens, 0, args.n_predict)
        probed = generate(server, case["prompt"], args.effort, args.max_tokens,
                          args.probe_every, args.n_predict)
        same = plain["reasoning"] == probed["reasoning"] and plain["content"] == probed["content"]
        identical.append(same)
        ok = [p for p in probed["probes"] if p.get("success")]
        failed = [p for p in probed["probes"] if not p.get("success")]
        client_ms += [p["client_ms"] for p in ok]
        server_ms += [p["probe"]["timings"]["total_ms"] for p in ok]

        print("{}: reasoning {} chars, identical with and without probes: {}".format(
            name, len(plain["reasoning"]), "YES" if same else "NO"))
        print("  time {:.1f} s without probes, {:.1f} s with {} probes ({} failed)".format(
            plain["seconds"], probed["seconds"], len(ok), len(failed)))
        for p in failed[:3]:
            print("    failed at {} chars: {}".format(p["at_chars"], p.get("message")))
        for p in ok:
            t = p["probe"]["timings"]
            print("    at {:>5} chars: {:>8} p={:.3f}  client {:6.1f} ms  server {:6.1f} ms "
                  "(save {:.1f}, decode {:.1f}, restore {:.1f}, state {:.1f} MiB)".format(
                      p["at_chars"], repr(p["probe"]["content"]), math.exp(p["probe"]["logprob"]),
                      p["client_ms"], t["total_ms"], t["save_ms"], t["decode_ms"],
                      t["restore_ms"], t["state_mib"]))
        if not same:
            a, b = plain["reasoning"] + plain["content"], probed["reasoning"] + probed["content"]
            where = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print("  first difference at character {}: {!r} vs {!r}".format(
                where, a[where:where + 60], b[where:where + 60]))
        if out:
            out.write(json.dumps({"case": name, "plain": plain, "probed": probed}, ensure_ascii=False) + "\n")
            out.flush()

    if client_ms:
        print()
        print("{} probes: client median {:.0f} ms (p90 {:.0f}), server median {:.0f} ms; "
              "identical generations {}/{}".format(
                  len(client_ms), statistics.median(client_ms),
                  sorted(client_ms)[int(0.9 * (len(client_ms) - 1))],
                  statistics.median(server_ms), sum(identical), len(identical)))
    return 0 if all(identical) else 1


if __name__ == "__main__":
    raise SystemExit(main())
