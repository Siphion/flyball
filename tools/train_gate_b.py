#!/usr/bin/env python3
"""Train, calibrate and judge the Gate B governor.

Input: uncut reasoning traces, the cases they came from, and the teacher's labels
from probe_traces.py. Output: a JSON model and a report that answers one question:

    at the same accuracy, does the learned governor save more reasoning than a
    fixed reasoning budget?

That is the bar. A governor that cannot clear it is a fixed budget with extra
steps, and `--reasoning-budget` already exists.

Problems are split three ways by a hash of their name, never by checkpoint: every
checkpoint of a problem lands in the same split, so the test set contains no
problem the model has seen. Training fits the model, the calibration split fits the
mapping from score to probability and chooses the configuration, and the test split
is only ever evaluated.

Several datasets can be given at once (repeat the three options in the same order):
the governor is meant to face easy and hard problems alike, and a mixture is where a
fixed budget should struggle.

When the probes come from the `trial` protocol they also carry the model's own
confidence in its forced answer, and two self-probing governors are simulated next
to the learned one: cut when the model is confident enough, or when it gives the
same answer at two checkpoints in a row. They need the big model at every check, so
they are reported with that cost; the learned governor runs on the CPU for free.

Usage:
    python tools/train_gate_b.py \\
        --traces reports/gsm8k_traces.jsonl --cases reports/gsm8k_cases.json \\
        --probes reports/gsm8k_trial_probes.jsonl \\
        --traces reports/math_traces.jsonl --cases reports/math_cases.json \\
        --probes reports/math_probes.jsonl \\
        --model-out reports/gate_b_model.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flyball.gate_b import (  # noqa: E402
    DENSE_NAMES,
    dense_features,
    fit_logistic,
    fit_platt,
    vectorise,
)
from measure_reasoning_cut import last_number  # noqa: E402

STATIC_BUDGETS = [200, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
CONFIDENCES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def split_of(case: str) -> str:
    bucket = int(hashlib.sha256(case.encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else ("calib" if bucket < 8 else "test")


def load_rows(traces_path: Path, cases_path: Path, probes_path: Path) -> list[dict]:
    cases = {c["name"]: c for c in json.loads(cases_path.read_text(encoding="utf-8"))}
    traces = {}
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            if not t.get("cut"):
                traces[(t["case"], t["repeat"])] = t
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # A probe at the very end of the reasoning (probe_traces.py --final) grades the
    # uncut trace the same way every cut is graded. Without one, the grade read from
    # the reply at generation time stands.
    final = {(p["case"], p["repeat"]): p["safe"] for p in probes if p.get("final")}
    rows = []
    for p in probes:
        trace = traces.get((p["case"], p["repeat"]))
        if trace is None or p.get("final"):
            continue
        rows.append({
            "case": p["case"],
            "repeat": p["repeat"],
            "dataset": cases[p["case"]].get("class", "?").split("-")[0],
            "position": p["position"],
            "length": len(trace["reasoning"]),
            "total_chars": len(trace["reasoning"]) + len(trace["content"]),
            "seconds": trace["seconds"],
            "safe": bool(p["safe"]),
            "trace_correct": bool(final.get((p["case"], p["repeat"]), trace["correct"])),
            "prompt": cases[p["case"]]["prompt"],
            "prefix": trace["reasoning"][: p["position"]],
            "split": split_of(p["case"]),
            "probe_seconds": p.get("probe_seconds", 0.0),
            # Present only with the trial protocol.
            "confidence": math.exp(p["logprob"]) if "logprob" in p else None,
            "trial_answer": last_number(p["trial"]) if "trial" in p else None,
        })
    return rows


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - labels[mask].mean())
    return total


def simulate(traces: dict, decide, probe_cost: bool = False) -> tuple[float, float, float, float]:
    """Walk each trace's checkpoints in order; `decide(row)` says cut or not.

    Returns accuracy, mean share of reasoning skipped, mean checks per trace, and the
    share of generation time saved. Time saved is estimated from the characters
    skipped, minus the probes' own time when the policy needs the big model to check.
    """
    correct, skipped, checks, saved, spent = [], [], [], 0.0, 0.0
    for rows in traces.values():
        rows = sorted(rows, key=lambda r: r["position"])
        index = next((i for i, r in enumerate(rows) if decide(r)), None)
        used = rows if index is None else rows[: index + 1]
        checks.append(len(used))
        seconds = rows[0]["seconds"]
        spent += seconds
        cost = sum(r["probe_seconds"] for r in used) if probe_cost else 0.0
        if index is None:
            correct.append(rows[0]["trace_correct"])
            skipped.append(0.0)
            saved -= cost
        else:
            cut = rows[index]
            correct.append(cut["safe"])
            skipped.append(1 - cut["position"] / cut["length"])
            saved += seconds * (cut["length"] - cut["position"]) / cut["total_chars"] - cost
    return (float(np.mean(correct)), float(np.mean(skipped)), float(np.mean(checks)),
            saved / spent if spent else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, action="append", required=True)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--probes", type=Path, action="append", required=True)
    parser.add_argument("--model-out", type=Path, default=Path("reports/gate_b_model.json"))
    args = parser.parse_args()
    if not len(args.traces) == len(args.cases) == len(args.probes):
        parser.error("give --traces, --cases and --probes the same number of times")

    rows = []
    for traces, cases, probes in zip(args.traces, args.cases, args.probes):
        rows += load_rows(traces, cases, probes)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    print("checkpoints: " + ", ".join(
        "{} {} ({} problems, {:.0%} safe)".format(
            name, len(by_split[name]), len({r["case"] for r in by_split[name]}),
            np.mean([r["safe"] for r in by_split[name]]) if by_split[name] else 0)
        for name in ("train", "calib", "test")))

    def matrix(split):
        x = np.stack([vectorise(r["prompt"], r["prefix"]) for r in by_split[split]])
        y = np.array([1.0 if r["safe"] else 0.0 for r in by_split[split]])
        return x, y

    x_train, y_train = matrix("train")
    x_calib, y_calib = matrix("calib")
    x_test, y_test = matrix("test")

    # Model selection uses the calibration split only. The test split is looked at
    # once, at the end; choosing anything on it would make its numbers optimistic.
    dense = len(DENSE_NAMES)
    word_mask = np.zeros(x_train.shape[1])
    word_mask[:dense] = 1.0

    candidates = []
    for words in (False, True):
        for l2 in (0.1, 1.0, 10.0, 100.0):
            mask = np.ones(x_train.shape[1]) if words else word_mask
            candidate = fit_logistic(x_train * mask, y_train, l2=l2)
            score = auc(candidate.raw_score(x_calib * mask), y_calib)
            candidates.append((score, words, l2, candidate, mask))
            print("  selection on calibration split: words={!s:5} l2={:<6} AUC {:.3f}".format(
                words, l2, score))
    _, words, l2, model, mask = max(candidates, key=lambda c: c[0])
    print("  chosen: words={} l2={}".format(words, l2))
    if not words:
        # Dense-only: the word weights are exactly zero, so the saved model does
        # not depend on the hashed features at all.
        model.weights = model.weights * mask

    fit_platt(model, x_calib * mask, y_calib)
    prob = model.probability(x_test * mask)
    model.save(args.model_out)

    print()
    print("Classifier on held-out problems")
    print("  AUC {:.3f}   ECE {:.3f}   accuracy at 0.5 {:.3f}   base rate {:.3f}".format(
        auc(prob, y_test), ece(prob, y_test), float(np.mean((prob >= 0.5) == (y_test == 1))),
        float(np.mean(y_test))))

    names = DENSE_NAMES
    top = np.argsort(-np.abs(model.weights[: len(names)]))[:5]
    print("  strongest hand-made features: " + ", ".join(
        "{} ({:+.2f})".format(names[i], model.weights[i]) for i in top))

    def traces_of(split: str, probabilities: np.ndarray) -> dict:
        grouped = defaultdict(list)
        for row, p in zip(by_split[split], probabilities):
            row["p"] = float(p)
            features = dense_features(row["prompt"], row["prefix"])
            row["concluded"] = features["has_conclusion"] > 0
            row["verifying"] = features["verifying_after_conclusion"] > 0
            grouped[(row["case"], row["repeat"])].append(row)
        for trace_rows in grouped.values():
            trace_rows.sort(key=lambda r: r["position"])
            previous = None
            for row in trace_rows:
                row["repeated"] = row["trial_answer"] is not None and row["trial_answer"] == previous
                previous = row["trial_answer"]
        return grouped

    calib_traces = traces_of("calib", model.probability(x_calib * mask))
    test_traces = traces_of("test", prob)
    trial = all(r["confidence"] is not None for r in rows)

    # Every policy, as (family, name, decide, needs the big model at each check).
    policies = [
        ("reference", "never cut (today)", lambda r: False, False),
        ("reference", "oracle: first safe checkpoint", lambda r: r["safe"], False),
        ("heuristic", "heuristic: a conclusion stated", lambda r: r["concluded"], False),
        ("heuristic", "heuristic: concluded, then verifying", lambda r: r["verifying"], False),
    ]
    policies += [("static budget", "static budget {} chars".format(b),
                  lambda r, b=b: r["position"] >= b, False) for b in STATIC_BUDGETS]
    policies += [("learned (CPU)", "learned (CPU), p >= {:.2f}".format(t),
                  lambda r, t=t: r["p"] >= t, False) for t in THRESHOLDS]
    if trial:
        policies += [("self-probe (GPU)", "self-probe (GPU), confidence >= {:.2f}".format(c),
                      lambda r, c=c: r["confidence"] >= c, True) for c in CONFIDENCES]
        policies += [("self-probe (GPU)", "self-probe (GPU), same answer twice",
                      lambda r: r["repeated"], True)]

    def report(title: str, keep) -> None:
        calib = {k: v for k, v in calib_traces.items() if keep(v[0])}
        test = {k: v for k, v in test_traces.items() if keep(v[0])}
        if not test:
            return
        print()
        print("{} ({} held-out traces)".format(title, len(test)))
        print("  {:38} {:>9} {:>14} {:>8} {:>11}".format(
            "policy", "accuracy", "reasoning cut", "checks", "time saved"))
        results = {}
        for family, name, decide, cost in policies:
            results[name] = simulate(test, decide, probe_cost=cost)
            print("  {:38} {:>8.1%} {:>13.0%} {:>8.1f} {:>10.0%}".format(name, *results[name]))

        # The operating point of each family is chosen on the calibration split, as
        # it would be before deployment, and only then read on the test split.
        print("  Operating point chosen on the calibration split (the most reasoning cut")
        print("  while staying within one point of never cutting), read on the test split:")
        floor = simulate(calib, lambda r: False)[0] - 0.01
        never = results["never cut (today)"]
        print("    {:17} accuracy {:.1%}".format("never cut", never[0]))
        for family in ("static budget", "learned (CPU)", "self-probe (GPU)"):
            members = [(name, decide, cost) for f, name, decide, cost in policies if f == family]
            if not members:
                continue
            eligible = []
            for name, decide, cost in members:
                accuracy, cut, _, _ = simulate(calib, decide, probe_cost=cost)
                if accuracy >= floor:
                    eligible.append((cut, name))
            if not eligible:
                print("    {:17} no setting keeps accuracy on the calibration split".format(family))
                continue
            _, name = max(eligible)
            accuracy, cut, checks, saved = results[name]
            print("    {:17} accuracy {:.1%}, reasoning cut {:.0%}, time saved {:.0%}  [{}]".format(
                family, accuracy, cut, saved, name))

    report("Governor simulation, all datasets", lambda r: True)
    datasets = sorted({r["dataset"] for r in by_split["test"]})
    if len(datasets) > 1:
        for dataset in datasets:
            report("Governor simulation, " + dataset, lambda r, d=dataset: r["dataset"] == d)
    print()
    print("model written to " + str(args.model_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
