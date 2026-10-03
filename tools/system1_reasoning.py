#!/usr/bin/env python3
"""Can a System 1 read, from the reasoning alone, that the model has its answer?

The probe asks the model itself, which costs a pause of the generation and needs a
server that can fork a live completion. A small decision model reading the
reasoning as it streams would need neither: it runs on the CPU, beside the GPU,
and on any server. A bag-of-words classifier failed at this (gate-b-research.md,
section 4); this tool asks a real decision model.

The backend is whatever speaks the System One protocol: Laya in this process today,
a Laya server or Jev's API tomorrow (flyball.backends). Every labelled checkpoint
becomes a state, the question and the end of the reasoning so far, and the backend
answers yes-or-no questions about it in one pass. Three phrasings are asked, because
a zero-shot decision model is sensitive to wording; the first one is decided before
looking at the results.

Results are cached, so a run can be interrupted and resumed. The report compares
each phrasing, and the model's own confidence on the same checkpoints, against the
teacher's label: does the answer at this checkpoint come out right?

Usage:
    python tools/system1_reasoning.py --backend laya --name math \\
        --traces reports/math_traces.jsonl --cases reports/math_cases.json \\
        --probes reports/math_probes.jsonl --sample 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flyball.backends import backend_from_spec  # noqa: E402

QUESTIONS = {
    "settled": {"type": "noul", "instructions":
                "Has this reasoning already reached its final answer, so that thinking any longer is unnecessary?"},
    "answer_stated": {"type": "noul", "instructions":
                      "Does the reasoning so far already state the final answer to the question?"},
    "still_working": {"type": "noul", "instructions":
                      "Is the reasoning still in the middle of working out the answer?"},
}
PRIMARY = "settled"


def state_of(prompt: str, prefix: str, question_chars: int, tail_chars: int) -> str:
    """The question, then the end of the reasoning so far: what a governor sees."""
    question = prompt if len(prompt) <= question_chars else prompt[:question_chars] + " [...]"
    tail = prefix if len(prefix) <= tail_chars else "[...] " + prefix[-tail_chars:]
    return "Question:\n" + question + "\n\nReasoning so far:\n" + tail


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="laya", help="laya, laya:<checkpoint>, or a System One URL")
    parser.add_argument("--name", required=True, help="a label for the dataset, used in file names")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0, help="checkpoints to sample, 0 for all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--question-chars", type=int, default=500)
    parser.add_argument("--tail-chars", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for a local backend")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    cases = {c["name"]: c for c in json.loads(args.cases.read_text(encoding="utf-8"))}
    traces = {(t["case"], t["repeat"]): t for t in map(json.loads, args.traces.read_text(encoding="utf-8").splitlines()) if not t.get("cut")}
    probes = [p for p in map(json.loads, args.probes.read_text(encoding="utf-8").splitlines())
              if not p.get("final") and (p["case"], p["repeat"]) in traces]
    if args.sample and args.sample < len(probes):
        probes = random.Random(args.seed).sample(probes, args.sample)

    label = args.backend.replace(":", "-").replace("/", "_")[:40]
    cache_path = args.out_dir / "system1_{}_{}.jsonl".format(label, args.name)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["case"], row["repeat"], row["position"])] = row

    backend = backend_from_spec(args.backend)
    if args.threads and hasattr(backend, "threads"):
        backend.threads = args.threads
    todo = [p for p in probes if (p["case"], p["repeat"], p["position"]) not in done]
    print("{}: {} checkpoints, {} cached, {} to ask {}".format(args.name, len(probes), len(probes) - len(todo), len(todo), args.backend))

    with cache_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            states = [state_of(cases[p["case"]]["prompt"], traces[(p["case"], p["repeat"])]["reasoning"][: p["position"]],
                               args.question_chars, args.tail_chars) for p in chunk]
            began = time.time()
            answers = backend.decide(states, QUESTIONS)
            each = (time.time() - began) / len(chunk)
            for p, a in zip(chunk, answers):
                row = {"case": p["case"], "repeat": p["repeat"], "position": p["position"],
                       "seconds": round(each, 3), **{q: round(float(a[q]["noul"]), 4) for q in QUESTIONS}}
                done[(p["case"], p["repeat"], p["position"])] = row
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            print("  {}/{} ({:.2f} s per checkpoint)".format(min(start + args.batch, len(todo)), len(todo), each), flush=True)

    # The report: every phrasing, and the model's own confidence, on the same checkpoints.
    rows = [(p, done[(p["case"], p["repeat"], p["position"])]) for p in probes]
    labels = np.array([1.0 if p["safe"] else 0.0 for p, _ in rows])
    print()
    print("{}: {} checkpoints, {:.0%} with the right answer already in place".format(args.name, len(rows), labels.mean()))
    print("  {:28} {:>6}".format("signal", "AUC"))
    for q in QUESTIONS:
        scores = np.array([r[q] for _, r in rows])
        if q == "still_working":
            scores = 1 - scores
        print("  {:28} {:>6.3f}{}".format("System 1: " + q, auc(scores, labels), "   (chosen beforehand)" if q == PRIMARY else ""))
    own = []
    for p, _ in rows:
        own.append(math.exp(p["logprob"]) if "logprob" in p else p.get("readiness", float("nan")))
    own = np.array(own)
    if not np.isnan(own).all():
        print("  {:28} {:>6.3f}".format("the model's own confidence", auc(own, labels)))
    seconds = [r["seconds"] for _, r in rows if "seconds" in r]
    if seconds:
        print("  {:.2f} s per checkpoint on this machine".format(float(np.median(seconds))))

    # The same checkpoints with the System 1's answer as their confidence, for
    # compare_governors.py: a decision on the CPU costs the generation nothing.
    derived = args.out_dir / "system1_{}_{}_probes.jsonl".format(label, args.name)
    with derived.open("w", encoding="utf-8") as handle:
        for p, r in rows:
            q = {k: v for k, v in p.items() if k not in ("logprob", "min_prob", "trial")}
            q["readiness"] = r[PRIMARY]
            q["probe_seconds"] = 0.0
            handle.write(json.dumps(q, ensure_ascii=False) + "\n")
        # The probes at the very end of each reasoning grade "never cut"; they carry over.
        for line in args.probes.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("final"):
                handle.write(line + "\n")
    print("  governor input written to " + str(derived))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
