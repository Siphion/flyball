#!/usr/bin/env python3
"""Where does the answer first appear in the reasoning, and can a cheap signal tell?

This is the offline research behind Gate B, the mid-stream governor. It needs no
proxy and no live model: only full reasoning traces of uncut runs, written by
`measure_reasoning_cut.py --traces`, and a strict `answer` pattern per case.

Two questions, answered in order.

1. The oracle. For each trace, the first character at which the correct final
   answer is already written in the reasoning. Cutting there is the best any
   governor could do; everything after it is reasoning the answer did not need.
   If this number is not much smaller than the full reasoning, there is nothing to
   govern, and the project has no thesis.

2. Generic signals. A real governor does not know the answer. It sees the reasoning
   so far and nothing else. A signal is useful only if it fires after the oracle
   point and rarely before: firing before the answer exists means cutting a problem
   that still needed thought, which is how the cryptarithm loses its answer.

Usage:
    python tools/analyze_traces.py --traces reports/m4_traces.jsonl \\
        --cases benchmarks/cases.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

CHECKPOINT_CHARS = 150

# Generic signals. None of them knows the answer; each is a guess at what
# "has thought enough" looks like from the outside.
CONCLUSION = re.compile(
    r"(?i)\b(so|thus|therefore|hence)\b[^.\n]{0,80}\b(answer|result|they meet|is|=|equals)\b"
    r"|\bfinal answer\b|\bthe answer is\b|\bboxed\b"
)
VERIFICATION = re.compile(
    r"(?i)\b(let me|let's|let us)\s+(double.?check|verify|check|confirm|re-?check)\b"
    r"|\bwait\b|\bactually\b|\bhmm\b|\bon second thought\b|\balternatively\b"
)
NUMBER = re.compile(r"\d+(?:[.:/]\d+)?")


def first_answer_end(reasoning: str, pattern: re.Pattern) -> int | None:
    match = pattern.search(reasoning)
    return match.end() if match else None


def repeated_ngram_share(text: str, window: str, n: int = 6) -> float:
    """Share of word n-grams in `window` that already occurred earlier in `text`."""
    words = text.split()
    tail = window.split()
    if len(tail) < n:
        return 0.0
    seen = {tuple(words[i : i + n]) for i in range(max(0, len(words) - len(tail) - n))}
    grams = [tuple(tail[i : i + n]) for i in range(len(tail) - n + 1)]
    return sum(1 for g in grams if g in seen) / len(grams)


def signals_at(reasoning: str, upto: int) -> dict:
    """Evaluate every generic signal on the prefix a governor would have seen."""
    prefix = reasoning[:upto]
    window = prefix[-CHECKPOINT_CHARS * 2 :]
    conclusions = [m.end() for m in CONCLUSION.finditer(prefix)]
    after_conclusion = prefix[conclusions[-1] :] if conclusions else ""
    earlier_numbers = set(NUMBER.findall(prefix[: -len(window)])) if len(prefix) > len(window) else set()
    window_numbers = set(NUMBER.findall(window))
    return {
        # A conclusion has been stated and the model is now checking it.
        "concluded_then_verifying": bool(conclusions) and bool(VERIFICATION.search(after_conclusion)),
        # A conclusion has been stated at all.
        "concluded": bool(conclusions),
        # The latest stretch mostly repeats what was already written.
        "repeating": repeated_ngram_share(prefix, window) >= 0.3,
        # The latest stretch introduces no number that was not seen before.
        "no_new_numbers": bool(earlier_numbers) and not (window_numbers - earlier_numbers),
    }


def analyse(traces: list[dict], cases: dict) -> None:
    by_case = defaultdict(list)
    for trace in traces:
        case = cases.get(trace["case"])
        if trace.get("cut") or not case or not case.get("answer"):
            continue
        by_case[trace["case"]].append(trace)

    if not by_case:
        print("No uncut traces with a strict answer pattern.")
        return

    print("1. Oracle: where the correct answer first appears in the reasoning")
    print()
    print("{:24} {:>3} {:>9} {:>10} {:>9} {:>9}".format(
        "case", "n", "reasoning", "answer at", "unneeded", "correct"))
    oracle_rows = {}
    for name in sorted(by_case):
        pattern = re.compile(cases[name]["answer"], re.I)
        rows = []
        for trace in by_case[name]:
            total = len(trace["reasoning"])
            end = first_answer_end(trace["reasoning"], pattern)
            rows.append((total, end, trace["correct"]))
        oracle_rows[name] = rows
        totals = [t for t, _, _ in rows]
        found = [(t, e) for t, e, _ in rows if e is not None]
        unneeded = [1 - e / t for t, e in found if t]
        print("{:24} {:>3} {:>9} {:>10} {:>9} {:>9}".format(
            name,
            len(rows),
            int(statistics.median(totals)),
            "{} ({}/{})".format(int(statistics.median([e for _, e in found])), len(found), len(rows))
            if found else "never",
            "{:.0%}".format(statistics.median(unneeded)) if unneeded else "-",
            "{}/{}".format(sum(1 for _, _, c in rows if c), len(rows)),
        ))

    print()
    print('"answer at" is the median first position of the correct answer, and in how many')
    print('traces it appears at all. "unneeded" is the median share of the reasoning written')
    print("after it. A case where the answer appears late or never must not be cut early.")

    print()
    print("2. Generic signals: do they fire after the answer, and not before?")
    print()
    names = ["concluded", "concluded_then_verifying", "repeating", "no_new_numbers"]
    print("{:26} {:>10} {:>10} {:>12} {:>10}".format(
        "signal", "premature", "on time", "saved chars", "missed"))
    for signal in names:
        premature = on_time = missed = 0
        saved = []
        for name, traces_ in by_case.items():
            pattern = re.compile(cases[name]["answer"], re.I)
            for trace in traces_:
                reasoning = trace["reasoning"]
                answer_end = first_answer_end(reasoning, pattern)
                fired_at = None
                for upto in range(CHECKPOINT_CHARS, len(reasoning) + 1, CHECKPOINT_CHARS):
                    if signals_at(reasoning, upto)[signal]:
                        fired_at = upto
                        break
                if fired_at is None:
                    missed += 1
                elif answer_end is None or fired_at < answer_end:
                    premature += 1
                else:
                    on_time += 1
                    saved.append(len(reasoning) - fired_at)
        print("{:26} {:>10} {:>10} {:>12} {:>10}".format(
            signal,
            premature,
            on_time,
            int(statistics.median(saved)) if saved else "-",
            missed,
        ))
    print()
    print("premature: the signal fired before the answer existed, so cutting there would")
    print("have removed reasoning the answer needed. on time: fired after it. saved chars:")
    print("median reasoning that would have been skipped when on time. missed: never fired.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline research for the mid-stream governor.")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases.json"))
    args = parser.parse_args()

    cases = {c["name"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}
    traces = [
        json.loads(line)
        for line in args.traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    analyse(traces, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
