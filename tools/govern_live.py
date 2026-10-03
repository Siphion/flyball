#!/usr/bin/env python3
"""Run the Gate B governor live and measure it end to end.

Everything before this was simulated: a probe's boxed answer stood in for the reply
the model writes after a real cut. Here the cuts are real. Each problem is answered
three times, by three arms:

- `never`: the reasoning runs to its natural end;
- `budget`: the reasoning is closed with `reasoning_end` once it reaches a fixed
  number of characters, which is what `--reasoning-budget` does;
- `governor`: the model is probed (the server's `probe` control action) once the
  reasoning passes 300 characters and each time it doubles; the reasoning is closed
  as soon as the model is confident enough in the answer it would give, and at a
  cap if that never happens.

The operating points are passed in, chosen beforehand on other problems: choosing
them here would make the result optimistic. The arms of a problem run back to back,
in an order rotated from one problem to the next, with prompt caching off, so that
neither drift nor a warm cache favours an arm.

Every reply is graded the same way, by the number in its final box. The whole texts
are kept, and the summary grades them again, so a fix to the grader reaches runs
already made: the first pass of this experiment kept only the last 300 characters
of each reply, and lost answers to a grader that read `12{,}000` as `000`. Time is
the whole generation, from request to last token, probes included.

Resumable: runs already in the output file are skipped.

Usage:
    python tools/govern_live.py --cases reports/live_cases.json \\
        --out reports/live_runs.jsonl --budget 2400 --confidence 0.8 --cap 3200
    python tools/govern_live.py --summary reports/live_runs.jsonl --cases reports/live_cases.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from measure_reasoning_cut import final_answer, last_number

PROBE_SUFFIX = "\n</think>\n\n**Final Answer**\n\n\\boxed"
ORDERS = [("never", "budget", "governor"), ("budget", "governor", "never"), ("governor", "never", "budget")]


class Server:
    def __init__(self, base_url: str, api_key: str, effort: str, max_tokens: int) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.effort = effort
        self.max_tokens = max_tokens

    def _post(self, path: str, body: dict, timeout: float):
        request = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), self.headers, method="POST"
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def control(self, body: dict) -> dict:
        try:
            with self._post("/v1/chat/completions/control", body, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - a failed control call is data, not a crash
            return {"success": False, "message": repr(exc)}

    def run(self, prompt: str, arm: str, budget: int, confidence: float, cap: int) -> dict:
        body = {
            "model": "default",
            "stream": True,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.effort,
            "reasoning_control": arm != "never",
            "cache_prompt": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        started = time.perf_counter()
        completion_id = None
        reasoning, content = [], []
        ttft = None
        cut = None
        probes = []
        next_probe = 300

        with self._post("/v1/chat/completions", body, timeout=1800) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                completion_id = completion_id or event.get("id")
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        content.append(delta["content"])
                if cut is not None or content or not completion_id:
                    continue

                chars = sum(len(part) for part in reasoning)
                reason = None
                if arm == "budget" and chars >= budget:
                    reason = "budget"
                elif arm == "governor":
                    if chars >= cap:
                        reason = "cap"
                    elif chars >= next_probe:
                        while next_probe <= chars:
                            next_probe *= 2
                        t0 = time.perf_counter()
                        reply = self.control({
                            "id": completion_id, "action": "probe", "suffix": PROBE_SUFFIX,
                            "n_predict": 12, "stop": "}",
                        })
                        probe = {"at": chars, "ms": round((time.perf_counter() - t0) * 1000, 1),
                                 "ok": bool(reply.get("success"))}
                        if probe["ok"]:
                            probe["answer"] = last_number(reply["probe"]["content"])
                            probe["p"] = round(math.exp(reply["probe"]["logprob"]), 4)
                            if probe["p"] >= confidence:
                                reason = "confident"
                        else:
                            probe["message"] = reply.get("message", "")
                        probes.append(probe)
                if reason:
                    ended = self.control({"id": completion_id, "action": "reasoning_end", "model": "default"})
                    cut = {"reason": reason, "at": chars, "ok": bool(ended.get("success")),
                           "at_s": round(time.perf_counter() - started, 2)}

        # The texts are kept whole: grading is redone from them at summary time, so a
        # fix to the grader applies to runs already made.
        return {
            "seconds": round(time.perf_counter() - started, 3),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "reasoning_chars": sum(len(part) for part in reasoning),
            "content_chars": sum(len(part) for part in content),
            "reasoning": "".join(reasoning),
            "content": "".join(content),
            "cut": cut,
            "probes": probes,
        }


def summarise(runs_path: Path, cases_path: Path, n_boot: int = 2000) -> None:
    cases = {c["name"]: c for c in json.loads(cases_path.read_text(encoding="utf-8"))}
    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {}
    for run in runs:
        if "content" in run:
            # graded again from the text, with the grader as it is now
            run["answer"] = final_answer(run["content"])
            run["correct"] = run["answer"] == last_number(cases[run["case"]]["reference_answer"])
        by.setdefault(run["case"], {})[run["arm"]] = run
    complete = [name for name, arms in by.items() if len(arms) == 3 and all("error" not in r for r in arms.values())]
    print("{} problems with all three arms".format(len(complete)))

    def table(names: list[str], title: str) -> None:
        if not names:
            return
        print()
        print("{} ({} problems)".format(title, len(names)))
        print("  {:9} {:>9} {:>14} {:>13} {:>12} {:>11} {:>8}".format(
            "arm", "accuracy", "mean seconds", "per request", "in total", "reasoning", "cuts"))
        never_t = np.array([by[n]["never"]["seconds"] for n in names])
        stats = {}
        for arm in ("never", "budget", "governor"):
            r = [by[n][arm] for n in names]
            t = np.array([x["seconds"] for x in r])
            correct = np.array([x["correct"] for x in r], dtype=float)
            per_request = float(np.mean((never_t - t) / never_t))
            in_total = float((never_t.sum() - t.sum()) / never_t.sum())
            cuts = sum(1 for x in r if x["cut"])
            stats[arm] = (correct, t)
            print("  {:9} {:>8.1%} {:>13.1f}s {:>12.0%} {:>12.0%} {:>11.0f} {:>8}".format(
                arm, correct.mean(), t.mean(), per_request, in_total,
                np.mean([x["reasoning_chars"] for x in r]), cuts))
        rng = np.random.default_rng(0)
        g_c, g_t = stats["governor"]
        b_c, b_t = stats["budget"]
        diffs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(names), len(names))
            diffs.append((
                g_c[i].mean() - b_c[i].mean(),
                np.mean((never_t[i] - g_t[i]) / never_t[i]) - np.mean((never_t[i] - b_t[i]) / never_t[i]),
                (b_t[i].sum() - g_t[i].sum()) / never_t[i].sum(),
            ))
        low, high = np.percentile(np.array(diffs), [2.5, 97.5], axis=0)
        print("  governor minus budget, 95% interval: accuracy [{:+.1%}, {:+.1%}]  "
              "time saved per request [{:+.0%}, {:+.0%}]  in total [{:+.0%}, {:+.0%}]".format(
                  low[0], high[0], low[1], high[1], low[2], high[2]))
        probes = [p for n in names for p in by[n]["governor"]["probes"]]
        if probes:
            ok = [p for p in probes if p["ok"]]
            print("  probes: {} ({} failed), {:.1f} per request, median {:.0f} ms".format(
                len(probes), len(probes) - len(ok), len(probes) / len(names),
                float(np.median([p["ms"] for p in ok])) if ok else float("nan")))
        reasons = {}
        for n in names:
            cut = by[n]["governor"]["cut"]
            key = cut["reason"] if cut else "no cut"
            reasons[key] = reasons.get(key, 0) + 1
        print("  governor cuts: " + ", ".join("{} {}".format(k, v) for k, v in sorted(reasons.items())))

    table(complete, "All problems")
    for prefix in ("math", "gsm8k"):
        table([n for n in complete if n.startswith(prefix)], prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path, help="only summarise an existing runs file")
    parser.add_argument("--only", help="run only the cases whose name starts with this")
    parser.add_argument("--upstream", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key", default=os.environ.get("FLYBALL_UPSTREAM_API_KEY", ""))
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--budget", type=int, default=2400, help="characters, for the budget arm")
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--cap", type=int, default=3200, help="characters, for the governor arm")
    args = parser.parse_args()

    if args.summary:
        summarise(args.summary, args.cases)
        return 0
    if not args.out:
        parser.error("--out is required to run")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["name"].startswith(args.only)]
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done.add((row["case"], row["arm"]))

    server = Server(args.upstream, args.api_key, args.effort, args.max_tokens)
    total = len(cases) * 3
    count = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            reference = last_number(case["reference_answer"])
            for arm in ORDERS[index % len(ORDERS)]:
                count += 1
                if (case["name"], arm) in done:
                    continue
                row = {"case": case["name"], "arm": arm, "budget": args.budget,
                       "confidence": args.confidence, "cap": args.cap}
                try:
                    result = server.run(case["prompt"], arm, args.budget, args.confidence, args.cap)
                    result["answer"] = final_answer(result["content"])
                    result["correct"] = result["answer"] == reference
                    row.update(result)
                except Exception as exc:  # noqa: BLE001 - a failed run is recorded and retried later
                    row["error"] = repr(exc)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print("[{:>3}/{}] {:<40} {:8} {:>6}s correct={} cut={}".format(
                    count, total, case["name"][:40], arm, row.get("seconds", "-"), row.get("correct"),
                    (row.get("cut") or {}).get("reason")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
