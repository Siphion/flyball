#!/usr/bin/env python3
"""Can a System 1 pick the reasoning level from the request alone? Measured.

Reads the labels of effort_ceiling.py (per problem: the probability of a right
answer, and the mean time, at each level) and one or more feature files of
embed_texts.py, and compares policies that choose a level per request:

- each fixed level, and the best fixed level of each benchmark (what a regex
  reading the prompt could do);
- the heuristic of flyball's first version;
- for each feature file, two learned System 1s: a logistic regression per level
  predicting the chance of a right answer, calibrated on held-out folds, and a
  kNN averaging the labels of the nearest training problems.

A learned policy steps down from the reference level (the level in use today):
it takes the cheapest level whose predicted chance of a right answer is within
`delta` of the reference level's. Everything is cross-validated by problem: five
outer folds; within each, the regularisation, the calibration and `delta` are
chosen on the other four folds only, `delta` as the largest saving that keeps
accuracy within --tolerance of the reference level there. The time a System 1
takes to decide is added to every request it decides.

With --artifact-out, the logistic System 1 of one feature file is refitted on all
problems and written as a Gate A artifact (flyball/gate_a.py) for the proxy.

Usage:
    python tools/gate_a_eval.py --labels reports/effort_labels.json \\
        --cases reports/live_cases.json --cases reports/mmlu_pro_cases_a.json \\
        --features state=F:/ai/flyball/research/gate_a_state.npz \\
        --features laya=F:/ai/flyball/research/gate_a_laya.npz \\
        --levels none,low,medium --reference medium
"""

from __future__ import annotations

import os

# A long fit on every core overheats this machine's CPU; four threads are plenty.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from evaluation import FOLDS, bootstrap, fold_of, interval, set_of  # noqa: E402

from flyball.calibration import fit_head, fit_platt, head_logits  # noqa: E402
from flyball.gate_a import GateAModel  # noqa: E402

L2S = (1e-3, 1e-2, 1e-1, 1.0)
DELTAS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


# -- policies -------------------------------------------------------------------


def relative_policy(p: np.ndarray, ref: int, delta: float) -> np.ndarray:
    """Per request, the cheapest level whose predicted chance is within `delta` of
    the reference level's. `p` is (requests, levels), cheapest level first."""
    ok = p >= (p[:, ref:ref + 1] - delta)
    return np.where(ok.any(axis=1), ok.argmax(axis=1), ref)


def outcome(P: np.ndarray, T: np.ndarray, choice: np.ndarray) -> tuple[float, float]:
    rows = np.arange(len(choice))
    return float(P[rows, choice].mean()), float(T[rows, choice].sum())


def choose_delta(p: np.ndarray, P: np.ndarray, T: np.ndarray, ref: int, tolerance: float) -> float:
    """The delta that saves the most time while keeping accuracy within `tolerance`
    of the reference level, on the data given (training folds only)."""
    ref_acc, ref_time = float(P[:, ref].mean()), float(T[:, ref].sum())
    best_delta, best_saving = 0.0, -np.inf
    for delta in DELTAS:
        acc, total = outcome(P, T, relative_policy(p, ref, delta))
        saving = 1 - total / ref_time
        if acc >= ref_acc - tolerance and saving > best_saving:
            best_delta, best_saving = delta, saving
    return best_delta


# -- learners -------------------------------------------------------------------


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logistic_logits(x_train, y_train, x_test, l2: float) -> np.ndarray:
    return head_logits(fit_head(x_train, y_train, l2=l2), x_test)


def inner_logits(x, y, folds, l2: float) -> np.ndarray:
    """Out-of-fold logits over the folds present in `folds`."""
    out = np.zeros(len(y))
    for g in np.unique(folds):
        test = folds == g
        out[test] = logistic_logits(x[~test], y[~test], x[test], l2)
    return out


def logistic_outer(x, P, folds, ref: int, T, tolerance: float) -> tuple[np.ndarray, list, list]:
    """Outer-fold calibrated predictions, the per-fold choice of delta, and the
    per-level regularisation chosen in each fold."""
    n, levels = P.shape
    pred = np.zeros((n, levels))
    deltas, l2s = [], []
    for f in range(FOLDS):
        train, test = folds != f, folds == f
        if not test.any():
            continue
        inner_p = np.zeros((train.sum(), levels))
        chosen = []
        for j in range(levels):
            y = P[train, j]
            best = min(L2S, key=lambda l2: log_loss(1 / (1 + np.exp(-inner_logits(x[train], y, folds[train], l2))), y))
            chosen.append(best)
            logits = inner_logits(x[train], y, folds[train], best)
            platt = fit_platt(logits, y)
            inner_p[:, j] = [platt.apply(v) for v in logits]
            outer_logits = logistic_logits(x[train], y, x[test], best)
            pred[test, j] = [platt.apply(v) for v in outer_logits]
        deltas.append(choose_delta(inner_p, P[train], T[train], ref, tolerance))
        l2s.append(chosen)
    return pred, deltas, l2s


def knn_predict(x_train, P_train, x_test, k: int) -> np.ndarray:
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    a = (x_train - mean) / scale
    b = (x_test - mean) / scale
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    nearest = np.argsort(-(b @ a.T), axis=1)[:, :k]
    return P_train[nearest].mean(axis=1)


def knn_outer(x, P, folds, ref: int, T, tolerance: float, k: int) -> tuple[np.ndarray, list]:
    pred = np.zeros(P.shape)
    deltas = []
    for f in range(FOLDS):
        train, test = folds != f, folds == f
        if not test.any():
            continue
        inner = np.zeros((train.sum(), P.shape[1]))
        tf = folds[train]
        for g in np.unique(tf):
            held = tf == g
            inner[held] = knn_predict(x[train][~held], P[train][~held], x[train][held], k)
        deltas.append(choose_delta(inner, P[train], T[train], ref, tolerance))
        pred[test] = knn_predict(x[train], P[train], x[test], k)
    return pred, deltas


def apply_deltas(pred: np.ndarray, folds: np.ndarray, deltas: list, ref: int) -> np.ndarray:
    choice = np.zeros(len(pred), dtype=int)
    present = [f for f in range(FOLDS) if (folds == f).any()]
    for f, delta in zip(present, deltas):
        rows = folds == f
        choice[rows] = relative_policy(pred[rows], ref, delta)
    return choice


# -- baselines ------------------------------------------------------------------


def per_set_best(P: np.ndarray, T: np.ndarray, sets: np.ndarray, tolerance: float) -> np.ndarray:
    choice = np.zeros(len(P), dtype=int)
    for name in np.unique(sets):
        rows = sets == name
        acc = P[rows].mean(axis=0)
        eligible = [j for j in range(P.shape[1]) if acc[j] >= acc.max() - tolerance]
        choice[rows] = min(eligible, key=lambda j: T[rows, j].sum())
    return choice


def heuristic_choice(prompts: list[str], levels: list[str], ref: int) -> np.ndarray:
    from flyball.heuristic import decide
    from flyball.state import build_state

    off = next((j for j, name in enumerate(levels) if name == "none"), ref)
    out = []
    for prompt in prompts:
        values = decide(build_state({"messages": [{"role": "user", "content": prompt}]})).values
        if not values["needs_deliberation"]:
            out.append(off)
        else:
            out.append(levels.index(values["effort"]) if values["effort"] in levels else ref)
    return np.array(out)


# -- report -----------------------------------------------------------------------


def report(policies: dict, P: np.ndarray, T: np.ndarray, sets: np.ndarray, levels: list[str],
           base: np.ndarray, n_boot: int) -> dict:
    out = {}
    groups = [("all", np.ones(len(P), dtype=bool))] + [(s, sets == s) for s in sorted(set(sets))]
    for title, rows in groups:
        print()
        print("{} ({} problems)".format(title, int(rows.sum())))
        print("  {:34} {:>9} {:>11} {:>11} {:>22}".format("policy", "accuracy", "total time", "vs best", "levels chosen"))
        b_acc, b_time = outcome(P[rows], T[rows], base[rows])
        for name, (choice, extra) in policies.items():
            acc, total = outcome(P[rows], T[rows], choice[rows])
            total += extra * rows.sum()
            counts = np.bincount(choice[rows], minlength=len(levels))
            print("  {:34} {:>8.1%} {:>10.0f}s {:>10.0%} {:>22}".format(
                name, acc, total, 1 - total / b_time,
                " ".join("{}:{}".format(levels[j][:3], c) for j, c in enumerate(counts) if c)))
            out.setdefault(name, {})[title] = {"accuracy": acc, "time": total, "saved_vs_best": 1 - total / b_time}
        if title == "all":
            idx = np.where(rows)[0]
            samples = bootstrap(len(idx), n_boot)
            for name, (choice, extra) in policies.items():
                if not name.startswith(("logistic", "knn")):
                    continue
                c = P[idx, choice[idx]]
                t = T[idx, choice[idx]] + extra
                bc, bt = P[idx, base[idx]], T[idx, base[idx]]
                acc = [c[i].mean() - bc[i].mean() for i in samples]
                saved = [1 - t[i].sum() / bt[i].sum() for i in samples]
                print("  {} minus best fixed per set, 95%: accuracy [{:+.1%}, {:+.1%}]  time saved [{:+.0%}, {:+.0%}]"
                      .format(name, *interval(acc), *interval(saved)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", action="append", default=[], help="NAME=PATH.npz from embed_texts.py")
    parser.add_argument("--cases", type=Path, action="append", default=[], help="for the heuristic baseline")
    parser.add_argument("--levels", default="none,low,medium", help="the ladder, cheapest first")
    parser.add_argument("--reference", default="medium")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--artifact-features", help="the NAME of --features to write the artifact from")
    parser.add_argument("--profile", default="bonsai-2")
    args = parser.parse_args()

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    ref = levels.index(args.reference)
    labels = {e["case"]: e for e in json.loads(args.labels.read_text(encoding="utf-8"))}
    names = sorted(n for n, e in labels.items() if all(lv in e["p_correct"] for lv in levels))
    feature_sets = {}
    for spec in args.features:
        name, _, path = spec.partition("=")
        data = np.load(path, allow_pickle=False)
        index = {n: i for i, n in enumerate(data["names"].tolist())}
        feature_sets[name] = (data, index, str(data["spec"]) if "spec" in data else name)
        names = [n for n in names if n in index]
    if not names:
        print("no problem has labels at every level and every feature set")
        return 1

    P = np.array([[labels[n]["p_correct"][lv] for lv in levels] for n in names])
    T = np.array([[labels[n]["seconds"][lv] for lv in levels] for n in names])
    sets = np.array([set_of(n) for n in names])
    folds = np.array([fold_of(n) for n in names])
    base = per_set_best(P, T, sets, args.tolerance)
    print("{} problems; levels {}; reference {}; best fixed level per set: {}".format(
        len(names), ",".join(levels), args.reference,
        ", ".join("{} {}".format(s, levels[base[sets == s][0]]) for s in sorted(set(sets)))))

    policies: dict = {}
    for j, level in enumerate(levels):
        policies["fixed " + level] = (np.full(len(names), j), 0.0)
    policies["best fixed per set"] = (base, 0.0)
    if args.cases:
        prompts = {}
        for path in args.cases:
            prompts.update({c["name"]: c["prompt"] for c in json.loads(path.read_text(encoding="utf-8"))})
        if all(n in prompts for n in names):
            policies["heuristic (first version)"] = (heuristic_choice([prompts[n] for n in names], levels, ref), 0.0)

    fitted = {}
    for name, (data, index, spec) in feature_sets.items():
        x = data["features"][[index[n] for n in names]].astype(np.float64)
        latency = float(np.median(data["latency"])) if "latency" in data else 0.0
        pred, deltas, l2s = logistic_outer(x, P, folds, ref, T, args.tolerance)
        policies["logistic " + name] = (apply_deltas(pred, folds, deltas, ref), latency)
        kpred, kdeltas = knn_outer(x, P, folds, ref, T, args.tolerance, args.knn)
        policies["knn " + name] = (apply_deltas(kpred, folds, kdeltas, ref), latency)
        fitted[name] = (x, pred, deltas, l2s, spec, latency)
        print("  {}: {} features, decision {:.0f} ms; delta per fold {}".format(
            name, x.shape[1], latency * 1000, deltas))

    numbers = report(policies, P, T, sets, levels, base, args.bootstrap)
    if args.report_json:
        args.report_json.write_text(json.dumps(numbers, indent=1), encoding="utf-8")

    if args.artifact_out:
        name = args.artifact_features or next(iter(fitted), None)
        if name not in fitted:
            parser.error("--artifact-features must name one of --features")
        x, pred, deltas, l2s, spec, latency = fitted[name]
        heads, platts = {}, {}
        for j, level in enumerate(levels):
            l2 = float(np.median([chosen[j] for chosen in l2s]))
            y = P[:, j]
            logits = inner_logits(x, y, folds, l2)
            platts[level] = fit_platt(logits, y)
            heads[level] = fit_head(x, y, l2=l2)
        model = GateAModel(profile=args.profile, levels=levels, reference_level=args.reference, features=spec,
                           heads=heads, platt=platts, delta=float(np.median(deltas)),
                           metrics={"cross_validated": numbers.get("logistic " + name, {}),
                                    "problems": len(names), "decision_seconds": latency})
        model.save(args.artifact_out)
        print("\nGate A artifact written to {} (features {}, delta {:.2f})".format(args.artifact_out, spec, model.delta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
