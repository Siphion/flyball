#!/usr/bin/env python3
"""Measure what closing the reasoning early actually costs and saves.

This is the experiment the README reports, generalised: several prompts, several cut
thresholds, N runs each, on your own machine and your own model. It writes a CSV, so
the numbers can be re-examined later instead of being remembered.

It deliberately measures *time to a complete answer*, not how much reasoning was
removed. Cutting the hidden thinking pushes part of it into the visible answer, so
reasoning length alone overstates the saving.

Usage:
    python tools/measure_reasoning_cut.py --upstream http://127.0.0.1:8080 \
        --model my-model --cuts 0,300,800 --runs 4 --out reports/cuts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path

# Each case carries an optional regex for the right answer, so correctness can be
# counted without a human in the loop. Keep the expectations unambiguous.
DEFAULT_CASES = [
    {
        "name": "two-trains",
        "prompt": (
            "A train leaves Milan at 09:00 at 120 km/h. Another leaves Rome at 09:30 "
            "at 150 km/h on the same 570 km line, heading towards it. At what time do "
            "they meet? Reason it through."
        ),
        "expect": r"\b11[:.]23\b",
    },
    {
        "name": "trivial-lookup",
        "prompt": "Reply with exactly the word OK and nothing else.",
        "expect": r"\bOK\b",
    },
]


@dataclass
class Run:
    case: str
    cut: int
    seconds: float
    reasoning_chars: int
    content_chars: int
    correct: bool | None
    control_ok: bool | None
    cut_fired_at_s: float | None = None
    error: str = ""


@dataclass
class Client:
    base_url: str
    api_key: str = ""
    model: str = "default"
    effort: str = "xhigh"
    max_tokens: int = 8192
    headers: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = "Bearer " + self.api_key

    def _post(self, path: str, body: dict, timeout: float = 900.0):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers=self.headers,
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def generate(self, prompt: str, cut_after: int | None):
        """Stream one completion, optionally closing the reasoning after N characters."""
        body = {
            "model": self.model,
            "stream": True,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.effort,
            "reasoning_control": bool(cut_after),
            "messages": [{"role": "user", "content": prompt}],
        }
        started = time.time()
        completion_id = ""
        reasoning: list[str] = []
        content: list[str] = []
        control_ok: bool | None = None
        fired_at: float | None = None

        response = self._post("/v1/chat/completions", body)
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            completion_id = event.get("id") or completion_id
            delta = (event.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if delta.get("content"):
                content.append(delta["content"])

            produced = sum(len(part) for part in reasoning)
            if cut_after and control_ok is None and completion_id and produced >= cut_after:
                fired_at = round(time.time() - started, 2)
                control_ok = self._end_reasoning(completion_id)

        elapsed = round(time.time() - started, 2)
        return elapsed, "".join(reasoning), "".join(content), control_ok, fired_at

    def _end_reasoning(self, completion_id: str) -> bool:
        """Close the reasoning block of a completion that is still streaming."""
        try:
            reply = self._post(
                "/v1/chat/completions/control",
                {"id": completion_id, "action": "reasoning_end", "model": self.model},
                timeout=30.0,
            )
            return json.loads(reply.read()).get("success") is True
        except urllib.error.HTTPError as exc:
            print("    control endpoint returned HTTP " + str(exc.code))
            return False
        except Exception as exc:  # noqa: BLE001 - a failed cut is data, not a crash
            print("    control endpoint failed: " + repr(exc))
            return False


def summarise(results: list[Run], cases: list[dict], cuts: list[int]) -> None:
    header = ("case", "cut", "median s", "spread", "answer ch", "correct")
    print()
    print("{:16} {:>5} {:>9} {:>13} {:>10} {}".format(*header))
    for case in cases:
        for cut in cuts:
            rows = [r for r in results if r.case == case["name"] and r.cut == cut and not r.error]
            if not rows:
                continue
            times = [r.seconds for r in rows]
            graded = [r for r in rows if r.correct is not None]
            score = "-"
            if graded:
                score = str(sum(1 for r in graded if r.correct)) + "/" + str(len(graded))
            spread = "{:.0f}-{:.0f}s".format(min(times), max(times))
            answer = int(statistics.mean([r.content_chars for r in rows]))
            print(
                "{:16} {:>5} {:>9.1f} {:>13} {:>10} {}".format(
                    case["name"], cut, statistics.median(times), spread, answer, score
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the cost and saving of cutting reasoning.")
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("FLYBALL_UPSTREAM_API_KEY", ""))
    parser.add_argument("--model", default="default")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument(
        "--cuts",
        default="0,300,800",
        help="comma-separated reasoning-character thresholds; 0 means do not cut",
    )
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--cases", type=Path, help="JSON file of {name, prompt, expect} objects")
    parser.add_argument("--out", type=Path, default=Path("reports/reasoning_cuts.csv"))
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else DEFAULT_CASES
    cuts = [int(value) for value in args.cuts.split(",")]
    client = Client(args.upstream, args.api_key, args.model, args.effort)

    results: list[Run] = []
    for case in cases:
        pattern = re.compile(case["expect"], re.I) if case.get("expect") else None
        for cut in cuts:
            for index in range(args.runs):
                label = "  {} cut={:<4} run {}/{}".format(
                    case["name"], cut or "none", index + 1, args.runs
                )
                try:
                    seconds, reasoning, content, control_ok, fired = client.generate(
                        case["prompt"], cut or None
                    )
                except Exception as exc:  # noqa: BLE001
                    print(label + "  FAILED: " + repr(exc))
                    results.append(Run(case["name"], cut, 0.0, 0, 0, None, None, error=repr(exc)))
                    continue
                correct = bool(pattern.search(content)) if pattern else None
                results.append(
                    Run(
                        case["name"],
                        cut,
                        seconds,
                        len(reasoning),
                        len(content),
                        correct,
                        control_ok,
                        fired,
                    )
                )
                print(
                    label
                    + "  {:6.1f}s  reasoning {:>5}  answer {:>5}  correct={}".format(
                        seconds, len(reasoning), len(content), correct
                    )
                )

    if not results:
        print("no results")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Run)])
        writer.writeheader()
        for row in results:
            writer.writerow(vars(row))

    summarise(results, cases, cuts)
    print()
    print("wrote " + str(args.out))
    print("A spread wider than the gap between two rows means the run count is too low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
