#!/usr/bin/env python3
"""Compare the governors that need no training, over every trace.

Static budgets and self-probing need no fitted model, only an operating point: a
budget, or a confidence threshold. So they need no test split either. Problems are
divided into five folds; the operating point is chosen on four of them and read on
the fifth, five times over, so that every trace is judged by a setting chosen
without it. The uncertainty comes from a bootstrap over problems.

Four metrics, because they disagree and each answers a different question:

- accuracy;
- reasoning cut: the share of each request's reasoning skipped, averaged over
  requests. How much shorter a typical request's thinking gets;
- time saved per request: the same, in generation time and net of the probes' own
  cost, averaged over requests. How much less a typical request makes you wait;
- time saved in total: generation time saved over all requests, divided by all
  generation time. Long requests dominate it, and the longest ones are the runaway
  traces that never reach an answer.

With `--learned`, the learned governor joins in. It needs training, so its scores
are cross-fitted: for each fold, a model is trained on three of the others,
calibrated on the fourth, and scores the fold it never saw. Every checkpoint then
has an out-of-fold score, and the operating point is chosen as for the other
families. A hybrid is judged too: the learned governor, which costs nothing, decides
when a probe is worth its price, and the probe decides when to cut.

Usage:
    python tools/compare_governors.py \\
        --traces reports/gsm8k_traces.jsonl --cases reports/gsm8k_cases.json \\
        --probes reports/gsm8k_trial_probes.jsonl \\
        --traces reports/math_traces.jsonl --cases reports/math_cases.json \\
        --probes reports/math_probes.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_gate_b import load_rows

FOLDS = 5
BUDGETS = [200, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 14400]
CONFIDENCES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
CAPS = [3200, 4800, 6400, 9600]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
# Sparse probing: once past 300 characters of reasoning, then each time it doubles.
DOUBLING = [300 * 2**k for k in range(10)]


def prepare(rows: list[dict]) -> dict:
    traces = defaultdict(list)
    for row in rows:
        traces[(row["case"], row["repeat"])].append(row)
    for trace in traces.values():
        trace.sort(key=lambda r: r["position"])
        seen = set()
        for row in trace:
            slot = max((g for g in DOUBLING if row["position"] >= g), default=None)
            row["doubling"] = slot is not None and slot not in seen
            if slot is not None:
                seen.add(slot)
    return traces


PROBE_COST = None  # None: the measured cost of each probe; (a, b): a + b seconds per 1000 chars


def probe_cost(row: dict) -> float:
    if PROBE_COST is None:
        return row["probe_seconds"]
    fixed, per_thousand = PROBE_COST
    return fixed + per_thousand * row["position"] / 1000


def outcome(trace: list[dict], decide, probed) -> tuple:
    """(correct, reasoning share skipped, seconds saved net of probes, seconds, probes)."""
    cost, probes = 0.0, 0
    for row in trace:
        if probed(row):
            cost += probe_cost(row)
            probes += 1
        if decide(row):
            saved = row["seconds"] * (row["length"] - row["position"]) / row["total_chars"]
            return row["safe"], 1 - row["position"] / row["length"], saved - cost, row["seconds"], probes
    return trace[0]["trace_correct"], 0.0, -cost, trace[0]["seconds"], probes


def metrics(outcomes) -> tuple:
    a = np.asarray(outcomes, dtype=float)
    return (a[:, 0].mean(), a[:, 1].mean(), (a[:, 2] / a[:, 3]).mean(),
            a[:, 2].sum() / a[:, 3].sum(), a[:, 4].mean())


def cross_fit(rows: list[dict], fold_of: dict, l2: float, words: bool) -> None:
    """Give every checkpoint an out-of-fold score from the learned governor."""
    from flyball.gate_b import DENSE_NAMES, fit_logistic, fit_platt, vectorise

    x = np.stack([vectorise(r["prompt"], r["prefix"]) for r in rows])
    y = np.array([1.0 if r["safe"] else 0.0 for r in rows])
    folds = np.array([fold_of[(r["case"], r["repeat"])] for r in rows])
    mask = np.ones(x.shape[1])
    if not words:
        mask[len(DENSE_NAMES):] = 0.0
    scores = np.zeros(len(rows))
    for held_out in range(FOLDS):
        calibration = (held_out + 1) % FOLDS
        train = (folds != held_out) & (folds != calibration)
        model = fit_logistic(x[train] * mask, y[train], l2=l2)
        fit_platt(model, x[folds == calibration] * mask, y[folds == calibration])
        scores[folds == held_out] = model.probability(x[folds == held_out] * mask)
    for row, score in zip(rows, scores):
        row["p"] = float(score)


def families(learned: bool) -> dict:
    never = lambda r: False  # noqa: E731
    extra = {}
    if learned:
        extra = {
            "learned (CPU)": [
                ("p {}".format(t), lambda r, t=t: r["p"] >= t, never) for t in THRESHOLDS],
            "learned gate + self-probe": [
                ("probe at p {}, cut at confidence {}".format(g, c),
                 lambda r, g=g, c=c: r["p"] >= g and r["confidence"] >= c,
                 lambda r, g=g: r["p"] >= g)
                for g in (0.5, 0.6, 0.7, 0.8) for c in CONFIDENCES],
        }
    return {
        "static budget": [
            ("{} chars".format(b), lambda r, b=b: r["position"] >= b, never) for b in BUDGETS],
        "self-probe, every checkpoint": [
            ("confidence {}".format(c), lambda r, c=c: r["confidence"] >= c, lambda r: True)
            for c in CONFIDENCES],
        "self-probe, doubling": [
            ("confidence {}".format(c), lambda r, c=c: r["doubling"] and r["confidence"] >= c,
             lambda r: r["doubling"]) for c in CONFIDENCES],
        "self-probe, doubling + cap": [
            ("confidence {} cap {}".format(c, cap),
             lambda r, c=c, cap=cap: (r["doubling"] and r["confidence"] >= c) or r["position"] >= cap,
             lambda r: r["doubling"]) for c in CONFIDENCES for cap in CAPS],
        **extra,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, action="append", required=True)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--probes", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--learned", action="store_true",
                        help="also judge the learned governor, cross-fitted (slower)")
    parser.add_argument("--l2", type=float, default=100.0, help="for --learned")
    parser.add_argument("--dense-only", action="store_true", help="for --learned: no word features")
    parser.add_argument(
        "--probe-cost",
        help="price each probe as 'a,b': a seconds plus b seconds per 1000 characters of "
        "reasoning, instead of the measured cost. The measured cost includes one request "
        "per answer token, which the probe needs only to read probabilities that "
        "speculative decoding hides",
    )
    args = parser.parse_args()
    if args.probe_cost:
        global PROBE_COST
        PROBE_COST = tuple(float(v) for v in args.probe_cost.split(","))
    if not len(args.traces) == len(args.cases) == len(args.probes):
        parser.error("give --traces, --cases and --probes the same number of times")

    rows = []
    for traces, cases, probes in zip(args.traces, args.cases, args.probes):
        rows += load_rows(traces, cases, probes)
    if any(r["confidence"] is None for r in rows):
        parser.error("self-probing needs probes from the trial protocol")
    traces = prepare(rows)
    keys = sorted(traces)
    fold = {k: int(hashlib.sha256(k[0].encode()).hexdigest(), 16) % FOLDS for k in keys}
    if args.learned:
        cross_fit(rows, fold, l2=args.l2, words=not args.dense_only)
    dataset = {k: traces[k][0]["dataset"] for k in keys}
    never = (lambda r: False, lambda r: False)

    baseline = metrics([outcome(traces[k], *never) for k in keys])
    runaway = sum(1 for k in keys if traces[k][0]["total_chars"] == traces[k][0]["length"])
    print("{} traces ({}), {} of them runaway (no answer before the token limit)".format(
        len(keys), ", ".join("{} {}".format(sum(1 for k in keys if dataset[k] == d), d)
                             for d in sorted(set(dataset.values()))), runaway))
    print("never cutting: accuracy {:.1%}".format(baseline[0]))

    header = "  {:30} {:>8} {:>10} {:>13} {:>11} {:>7}".format(
        "", "accuracy", "reasoning", "time saved", "time saved", "probes")
    subheader = "  {:30} {:>8} {:>10} {:>13} {:>11} {:>7}".format(
        "", "", "cut", "per request", "in total", "")

    for criterion in ("reasoning cut", "time saved in total"):
        index = 1 if criterion == "reasoning cut" else 3
        print()
        print("Operating point: the most {} within one point of never cutting,".format(criterion))
        print("chosen on four folds and read on the fifth.")
        print(header)
        print(subheader)
        judged = {}
        for family, members in families(args.learned).items():
            per_trace, chosen = {}, []
            for f in range(FOLDS):
                inside = [k for k in keys if fold[k] != f]
                floor = metrics([outcome(traces[k], *never) for k in inside])[0] - 0.01
                best = None
                for name, decide, probed in members:
                    result = metrics([outcome(traces[k], decide, probed) for k in inside])
                    if result[0] >= floor and (best is None or result[index] > best[0]):
                        best = (result[index], name, decide, probed)
                best = best or (0.0, "never", *never)
                chosen.append(best[1])
                for k in keys:
                    if fold[k] == f:
                        per_trace[k] = outcome(traces[k], best[2], best[3])
            judged[family] = per_trace
            print("  {:30} {:>7.1%} {:>10.0%} {:>13.0%} {:>11.0%} {:>7.1f}".format(
                family, *metrics(list(per_trace.values()))))
            for name in sorted(set(dataset.values())):
                print("    {:28} {:>7.1%} {:>10.0%} {:>13.0%} {:>11.0%} {:>7.1f}".format(
                    name, *metrics([per_trace[k] for k in keys if dataset[k] == name])))
            print("    settings chosen: " + ", ".join(sorted(set(chosen))))

        print("  Difference from the static budget, 95% bootstrap interval over problems:")
        rng = np.random.default_rng(0)
        static = np.array([judged["static budget"][k] for k in keys], dtype=float)
        samples = [rng.integers(0, len(keys), len(keys)) for _ in range(args.bootstrap)]
        for family, per_trace in judged.items():
            if family == "static budget":
                continue
            other = np.array([per_trace[k] for k in keys], dtype=float)
            diffs = np.array([np.subtract(metrics(other[s]), metrics(static[s])) for s in samples])
            low, high = np.percentile(diffs, [2.5, 97.5], axis=0)
            print("    {:28} accuracy [{:+.1%}, {:+.1%}]  reasoning cut [{:+.0%}, {:+.0%}]  "
                  "per request [{:+.0%}, {:+.0%}]  in total [{:+.0%}, {:+.0%}]".format(
                      family, low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
