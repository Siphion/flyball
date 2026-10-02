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
mapping from score to probability, and the test split is only ever evaluated.

Usage:
    python tools/train_gate_b.py --traces reports/gsm8k_traces.jsonl \\
        --cases reports/gsm8k_cases.json --probes reports/gsm8k_probes.jsonl \\
        --model-out reports/gate_b_model.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

STATIC_BUDGETS = [200, 400, 600, 800, 1200, 1600, 2400, 3200, 4800]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]


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
    rows = []
    for line in probes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        trace = traces.get((p["case"], p["repeat"]))
        if trace is None:
            continue
        rows.append({
            "case": p["case"],
            "repeat": p["repeat"],
            "position": p["position"],
            "length": len(trace["reasoning"]),
            "safe": bool(p["safe"]),
            "trace_correct": bool(trace["correct"]),
            "prompt": cases[p["case"]]["prompt"],
            "prefix": trace["reasoning"][: p["position"]],
            "split": split_of(p["case"]),
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


def simulate(traces: dict, decide) -> tuple[float, float]:
    """Walk each test trace's checkpoints in order; `decide(row)` says cut or not.
    Returns (accuracy, mean share of reasoning skipped)."""
    correct, skipped = [], []
    for rows in traces.values():
        rows = sorted(rows, key=lambda r: r["position"])
        cut = next((r for r in rows if decide(r)), None)
        if cut is None:
            correct.append(rows[0]["trace_correct"])
            skipped.append(0.0)
        else:
            correct.append(cut["safe"])
            skipped.append(1 - cut["position"] / cut["length"])
    return float(np.mean(correct)), float(np.mean(skipped))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("reports/gate_b_model.json"))
    parser.add_argument("--l2", type=float, default=50.0)
    args = parser.parse_args()

    rows = load_rows(args.traces, args.cases, args.probes)
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

    test_traces = defaultdict(list)
    for row, p in zip(by_split["test"], prob):
        row["p"] = float(p)
        dense = dense_features(row["prompt"], row["prefix"])
        row["concluded"] = dense["has_conclusion"] > 0
        row["verifying"] = dense["verifying_after_conclusion"] > 0
        test_traces[(row["case"], row["repeat"])].append(row)

    print()
    print("Governor simulation on {} held-out traces".format(len(test_traces)))
    print("  {:34} {:>9} {:>14}".format("policy", "accuracy", "reasoning cut"))

    def line(name, result):
        print("  {:34} {:>8.1%} {:>13.0%}".format(name, *result))
        return result

    never = line("never cut (today)", simulate(test_traces, lambda r: False))
    line("oracle: first safe checkpoint", simulate(test_traces, lambda r: r["safe"]))
    line("heuristic: a conclusion stated", simulate(test_traces, lambda r: r["concluded"]))
    line("heuristic: concluded, then verifying", simulate(test_traces, lambda r: r["verifying"]))
    static = {}
    for budget in STATIC_BUDGETS:
        static[budget] = line("static budget {} chars".format(budget),
                              simulate(test_traces, lambda r, b=budget: r["position"] >= b))
    learned = {}
    for threshold in THRESHOLDS:
        learned[threshold] = line("learned, cut when p >= {:.2f}".format(threshold),
                                  simulate(test_traces, lambda r, t=threshold: r["p"] >= t))

    print()
    print("At equal accuracy (within one point of never cutting):")
    floor = never[0] - 0.01
    best_static = max((v for v in static.values() if v[0] >= floor), key=lambda v: v[1], default=None)
    best_learned = max((v for v in learned.values() if v[0] >= floor), key=lambda v: v[1], default=None)
    print("  best static budget cuts   {}".format("{:.0%}".format(best_static[1]) if best_static else "nothing"))
    print("  best learned governor cuts {}".format("{:.0%}".format(best_learned[1]) if best_learned else "nothing"))
    print()
    print("model written to " + str(args.model_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
