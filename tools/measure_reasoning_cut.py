#!/usr/bin/env python3
"""Measure what closing the reasoning early actually costs and saves.

The question this answers is the one flyball exists to answer: for which kinds of
request does cutting the thinking pay, and where does it break the answer?

Three choices in here are deliberate.

*Paired, interleaved order.* The arms of one case run back to back, and the whole
set repeats. Measured variance on a local model is large — 21 to 33 seconds at
identical settings — so grouping all runs of one arm together would hand that
drift to whichever arm ran while the machine was busy.

*Time to a complete answer is the headline, time to first content token is the
companion.* Cutting the hidden thinking pushes part of it into the visible answer,
so reasoning length alone overstates the saving.

*Correctness is checked by regex where it can be.* A speed win that silently
breaks answers is not a win, and an unchecked arm invites wishful reading.

Usage:
    python tools/measure_reasoning_cut.py --cases benchmarks/cases.json \\
        --model my-model --cuts 0,300,800 --runs 8 --out reports/cuts.csv
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

DEFAULT_CASES = [
    {
        "name": "two-trains",
        "class": "math",
        "prompt": (
            "A train leaves Milan at 09:00 at 120 km/h. Another leaves Rome at 09:30 "
            "at 150 km/h on the same 570 km line, heading towards it. At what time do "
            "they meet? Reason it through."
        ),
        "expect": r"\b11[:.]23\b",
    },
    {
        "name": "trivial-ok",
        "class": "trivial",
        "prompt": "Reply with exactly the word OK and nothing else.",
        "expect": r"\bOK\b",
    },
]


@dataclass
class Run:
    case: str
    case_class: str
    cut: int
    repeat: int
    seconds: float
    ttft_s: float | None
    reasoning_chars: int
    content_chars: int
    correct: bool | None
    cut_applied: bool
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

    def generate(self, case: dict, cut_after: int | None) -> dict:
        """Stream one completion, optionally closing the reasoning after N characters."""
        body = {
            "model": self.model,
            "stream": True,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.effort,
            "reasoning_control": bool(cut_after),
            "messages": [{"role": "user", "content": case["prompt"]}],
        }
        if case.get("tools"):
            body["tools"] = case["tools"]
            body["tool_choice"] = "auto"

        started = time.time()
        completion_id = ""
        reasoning: list[str] = []
        content: list[str] = []
        tool_names: list[str] = []
        ttft: float | None = None
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
                if ttft is None:
                    ttft = round(time.time() - started, 2)
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name:
                    if ttft is None:
                        ttft = round(time.time() - started, 2)
                    tool_names.append(name)

            produced = sum(len(part) for part in reasoning)
            if cut_after and control_ok is None and completion_id and produced >= cut_after:
                fired_at = round(time.time() - started, 2)
                control_ok = self._end_reasoning(completion_id)

        return {
            "seconds": round(time.time() - started, 2),
            "ttft_s": ttft,
            "reasoning": "".join(reasoning),
            "content": "".join(content),
            "tool_names": tool_names,
            "control_ok": control_ok,
            "cut_fired_at_s": fired_at,
        }

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


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Thousands separators as LaTeX writes them: 12{,}000, and the thin spaces 12\,000 and
# 12\!000. Left in, they split one number into two, and 12{,}000 reads as 000.
_LATEX_SEPARATOR = re.compile(r"(?<=\d)(?:\{,\}|\\[,!;: ])(?=\d)")


def last_number(text: str) -> str | None:
    """The last number in a text, without thousands separators: the usual way to
    read a GSM8K answer, since the reference number also appears in the working."""
    found = _NUMBER.findall(_LATEX_SEPARATOR.sub("", text))
    if not found:
        return None
    value = found[-1].replace(",", "")
    # 12.00 and 12.0 are 12: money amounts are often written with cents.
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


BOX = r"\boxed{"


def final_answer(text: str) -> str | None:
    r"""The number a reply commits to: the content of the last \boxed{...} when there
    is one, otherwise the last number. Competition maths answers end in a box, and
    whatever follows the box (a check, a remark) must not be read as the answer."""
    start = text.rfind(BOX)
    if start >= 0:
        depth = 0
        opening = start + len(BOX) - 1
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    inside = last_number(text[opening + 1:index])
                    if inside is not None:
                        return inside
                    break
    return last_number(text)


def grade(case: dict, result: dict) -> bool | None:
    """None means this case has no machine-checkable expectation."""
    if case.get("expect_tool"):
        return case["expect_tool"] in result["tool_names"]
    if case.get("reference_answer") is not None:
        return final_answer(result["content"]) == last_number(case["reference_answer"])
    if case.get("expect"):
        return bool(re.search(case["expect"], result["content"], re.I))
    return None


def summarise(results: list[Run], by: str) -> None:
    """Aggregate by class or by case, so that the effect can be read per kind."""
    ok = [r for r in results if not r.error]
    keys = sorted({(getattr(r, by), r.cut) for r in ok})
    groups = sorted({k[0] for k in keys})
    cuts = sorted({k[1] for k in keys})

    print()
    print("{:16} {:>5} {:>4} {:>9} {:>12} {:>7} {:>9} {:>8}".format(
        by.replace("case_class", "class"), "cut", "n", "median s", "spread", "ttft", "answer ch", "correct"
    ))
    for group in groups:
        baseline = None
        for cut in cuts:
            rows = [r for r in ok if getattr(r, by) == group and r.cut == cut]
            if not rows:
                continue
            times = [r.seconds for r in rows]
            median = statistics.median(times)
            if cut == 0:
                baseline = median
            ttfts = [r.ttft_s for r in rows if r.ttft_s is not None]
            graded = [r for r in rows if r.correct is not None]
            score = "-"
            if graded:
                score = "{}/{}".format(sum(1 for r in graded if r.correct), len(graded))
            delta = ""
            if baseline and cut != 0:
                delta = "  {:+.0f}%".format(100.0 * (median - baseline) / baseline)
            print("{:16} {:>5} {:>4} {:>9.1f} {:>12} {:>7} {:>9} {:>8}{}".format(
                group, cut, len(rows), median,
                "{:.0f}-{:.0f}s".format(min(times), max(times)),
                "{:.1f}s".format(statistics.median(ttfts)) if ttfts else "-",
                int(statistics.mean([r.content_chars for r in rows])),
                score, delta,
            ))


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
    parser.add_argument("--runs", type=int, default=4, help="repeats of the whole interleaved set")
    parser.add_argument("--cases", type=Path, help="JSON file of case objects")
    parser.add_argument("--only-class", help="restrict to one class, for a second-stage sweep")
    parser.add_argument("--group-by", default="case_class", choices=["case_class", "case"])
    parser.add_argument("--out", type=Path, default=Path("reports/reasoning_cuts.csv"))
    parser.add_argument(
        "--traces",
        type=Path,
        help="also write every reasoning and answer in full, one JSON object per run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to existing output, skipping runs already in the traces file",
    )
    args = parser.parse_args()
    if args.resume and not args.traces:
        parser.error("--resume needs --traces, which records what has been done")

    cases = json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else DEFAULT_CASES
    if args.only_class:
        cases = [c for c in cases if c.get("class") == args.only_class]
        if not cases:
            print("no case matches class " + args.only_class)
            return 2
    cuts = [int(value) for value in args.cuts.split(",")]
    client = Client(args.upstream, args.api_key, args.model, args.effort)

    total = len(cases) * len(cuts) * args.runs
    print("{} cases x {} cuts x {} repeats = {} generations".format(
        len(cases), len(cuts), args.runs, total))

    results: list[Run] = []
    done = 0
    # Rows are flushed as they are produced: a sweep can run for hours, and an
    # interrupted run must keep everything it measured so far.
    # A resumed run skips what the traces file already holds. Runs that failed were
    # never written there, so they are attempted again.
    finished = set()
    if args.resume and args.traces.exists():
        for line in args.traces.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                finished.add((row["case"], row["cut"], row["repeat"]))
        print("resuming: {} runs already done".format(len(finished)))
    mode = "a" if args.resume else "w"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh_csv = mode == "w" or not args.out.exists()
    handle = args.out.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Run)])
    if fresh_csv:
        writer.writeheader()

    def record(run: Run) -> None:
        results.append(run)
        writer.writerow(vars(run))
        handle.flush()

    # Full texts go to a separate file: they are large, and they are what the
    # mid-stream research needs, where the CSV only carries lengths.
    traces = None
    if args.traces:
        args.traces.parent.mkdir(parents=True, exist_ok=True)
        traces = args.traces.open(mode, encoding="utf-8")

    # Interleaved: every arm of every case within one repeat, then repeat again.
    for repeat in range(args.runs):
        for case in cases:
            for cut in cuts:
                done += 1
                if (case["name"], cut, repeat) in finished:
                    continue
                label = "[{:>4}/{}] {:<22} cut={:<5}".format(
                    done, total, case["name"], cut or "none")
                try:
                    result = client.generate(case, cut or None)
                except Exception as exc:  # noqa: BLE001
                    print(label + " FAILED: " + repr(exc))
                    record(Run(
                        case["name"], case.get("class", "?"), cut, repeat,
                        0.0, None, 0, 0, None, False, None, error=repr(exc)))
                    continue
                correct = grade(case, result)
                record(Run(
                    case["name"], case.get("class", "?"), cut, repeat,
                    result["seconds"], result["ttft_s"],
                    len(result["reasoning"]), len(result["content"]),
                    correct, result["control_ok"] is True, result["control_ok"],
                    result["cut_fired_at_s"],
                ))
                if traces is not None:
                    traces.write(json.dumps({
                        "case": case["name"],
                        "case_class": case.get("class", "?"),
                        "cut": cut,
                        "repeat": repeat,
                        "correct": correct,
                        "seconds": result["seconds"],
                        "reasoning": result["reasoning"],
                        "content": result["content"],
                    }, ensure_ascii=False) + "\n")
                    traces.flush()
                print(label + " {:6.1f}s  ttft {:>5}  reasoning {:>5}  answer {:>5}  correct={}".format(
                    result["seconds"],
                    "{:.1f}s".format(result["ttft_s"]) if result["ttft_s"] else "-",
                    len(result["reasoning"]), len(result["content"]), correct))

    if not results:
        print("no results")
        return 1

    handle.close()
    if traces is not None:
        traces.close()

    summarise(results, args.group_by)
    never_fired = sorted({
        r.case for r in results if r.cut and not r.cut_applied and not r.error
    })
    if never_fired:
        print()
        print("The cut never fired for: " + ", ".join(never_fired))
        print("Those cases never produce that much reasoning, which is itself the finding:")
        print("a fixed threshold is inert on easy prompts.")
    print()
    print("wrote " + str(args.out))
    print("A spread wider than the gap between two rows means the repeat count is too low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
