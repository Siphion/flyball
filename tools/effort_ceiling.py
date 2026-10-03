#!/usr/bin/env python3
"""How much could choosing the reasoning level per problem save? The ceiling.

Gate A is worth building only if a perfect per-problem choice of the reasoning
level beats the best fixed level by a clear margin. This tool reads the runs of
effort_sweep.py, plus runs already made at one level (the never-cut arm of
govern_live.py, imported with --import PATH=LEVEL:REPEAT), grades every run from
its text, and reports per benchmark and overall:

- each fixed level: accuracy and time;
- the best fixed level: the cheapest whose accuracy is within --tolerance of the
  most accurate one, chosen per benchmark, as a regex reading the prompt could;
- the oracle on the same sample: for each problem, the fastest run that got it
  right. Optimistic twice over: it knows the answer, and with one noisy sample per
  cell it also picks lucky draws;
- the oracle on split samples: the cheapest level that was right on one repeat,
  judged on the other. What a perfect predictor of "this level is enough" could
  at best achieve, with the luck of the draw taken out;
- the same-level null: the same-sample oracle over two repeats of one level. Its
  gain is noise alone; the real same-sample oracle has to clear it by a margin.

With --labels-out it writes the labels a System 1 is trained on: per problem, the
probability of a right answer and the mean time at each level.

Usage:
    python tools/effort_ceiling.py --runs reports/effort_sweep.jsonl \\
        --cases reports/live_cases.json --cases reports/mmlu_pro_cases_a.json \\
        --import reports/live_medium_runs.jsonl=medium:0 \\
        --import reports/mmlu_pro_medium_runs.jsonl=medium:0 \\
        --levels none,low,medium --labels-out reports/effort_labels.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation import bootstrap, interval, set_of
from measure_reasoning_cut import answer_in, reference_of


def load_runs(runs: list[Path], imports: list[str], cases: dict) -> dict:
    """{(case, level, repeat): {"correct", "seconds"}}, every run graded again from
    its text. An imported file contributes its never-cut runs at the level given."""
    out: dict = {}

    def add(row: dict, level: str, repeat: int) -> None:
        case = cases.get(row["case"])
        if case is None or "error" in row or "content" not in row:
            return
        answer = answer_in(row["content"], case.get("answer_type", "number"))
        out[(row["case"], level, repeat)] = {"correct": answer is not None and answer == reference_of(case),
                                            "seconds": float(row["seconds"])}

    for path in runs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                add(row, row["level"], int(row["repeat"]))
    for spec in imports:
        path, _, target = spec.partition("=")
        level, _, repeat = target.partition(":")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("arm", "never") == "never":
                    add(row, level, int(repeat or 0))
    return out


class Table:
    """Correctness and time per problem, level and repeat, for one set of problems."""

    def __init__(self, runs: dict, names: list[str], levels: list[str], repeats: list[int]) -> None:
        self.names, self.levels, self.repeats = names, levels, repeats
        shape = (len(names), len(levels), len(repeats))
        self.c = np.zeros(shape)
        self.t = np.zeros(shape)
        for i, name in enumerate(names):
            for j, level in enumerate(levels):
                for k, repeat in enumerate(repeats):
                    run = runs[(name, level, repeat)]
                    self.c[i, j, k] = float(run["correct"])
                    self.t[i, j, k] = run["seconds"]

    def fixed(self, j: int, k: int | None = None, rows=None) -> tuple[float, float]:
        rows = slice(None) if rows is None else rows
        c = self.c[rows, j] if k is None else self.c[rows, j, k]
        t = self.t[rows, j] if k is None else self.t[rows, j, k]
        return float(c.mean()), float(t.sum() / (1 if k is not None else len(self.repeats)))


def complete_names(runs: dict, names: list[str], levels: list[str], repeats: list[int]) -> list[str]:
    return [n for n in names if all((n, lv, r) in runs for lv in levels for r in repeats)]


def best_fixed(table: Table, rows: np.ndarray, tolerance: float) -> int:
    """The cheapest level whose accuracy is within `tolerance` of the best."""
    stats = [table.fixed(j, rows=rows) for j in range(len(table.levels))]
    top = max(acc for acc, _ in stats)
    eligible = [j for j, (acc, _) in enumerate(stats) if acc >= top - tolerance]
    return min(eligible, key=lambda j: stats[j][1])


def per_set_best(table: Table, sets: np.ndarray, tolerance: float) -> np.ndarray:
    """For each problem, the best fixed level of its own benchmark."""
    choice = np.zeros(len(table.names), dtype=int)
    for name in np.unique(sets):
        rows = sets == name
        choice[rows] = best_fixed(table, rows, tolerance)
    return choice


def same_sample_oracle(c: np.ndarray, t: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Per problem, the fastest option that was right; the fallback when none was.
    `c` and `t` are (problems, options)."""
    timed = np.where(c > 0, t, np.inf)
    choice = timed.argmin(axis=1)
    return np.where(c.max(axis=1) > 0, choice, fallback)


def cheapest_right(c: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Per problem, the first level of the ladder (cheapest first) that was right."""
    right = c > 0
    first = right.argmax(axis=1)
    return np.where(right.any(axis=1), first, fallback)


def pick(values: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return values[np.arange(len(choice)), choice]


def ceiling(table: Table, sets: np.ndarray, tolerance: float, null_level: int) -> dict:
    """Every number of the report for one set of problems (all repeats used)."""
    base = per_set_best(table, sets, tolerance)
    out: dict = {"n": len(table.names), "fixed": {}, "per_set_best": [table.levels[j] for j in base]}
    for j, level in enumerate(table.levels):
        out["fixed"][level] = table.fixed(j)
    reps = range(len(table.repeats))

    # The baseline, per repeat: each problem at its benchmark's best fixed level.
    base_c = [pick(table.c[:, :, k], base) for k in reps]
    base_t = [pick(table.t[:, :, k], base) for k in reps]
    out["baseline"] = (float(np.mean(base_c)), float(np.mean([b.sum() for b in base_t])))

    same = [same_sample_oracle(table.c[:, :, k], table.t[:, :, k], base) for k in reps]
    out["same_sample"] = (float(np.mean([pick(table.c[:, :, k], same[k]).mean() for k in reps])),
                          float(np.mean([pick(table.t[:, :, k], same[k]).sum() for k in reps])))

    if len(table.repeats) >= 2:
        split_c, split_t, split_base_t, split_base_c = [], [], [], []
        for a, b in ((0, 1), (1, 0)):
            choice = cheapest_right(table.c[:, :, a], base)
            split_c.append(pick(table.c[:, :, b], choice))
            split_t.append(pick(table.t[:, :, b], choice))
            split_base_c.append(base_c[b])
            split_base_t.append(base_t[b])
        out["split_sample"] = (float(np.mean(split_c)), float(np.mean([s.sum() for s in split_t])))
        # Problem-level arrays for the intervals: averaged over the two directions.
        out["_split"] = (np.mean(split_c, axis=0), np.mean(split_t, axis=0),
                         np.mean(split_base_c, axis=0), np.mean(split_base_t, axis=0))

        # The null: two repeats of one level, as if they were two levels.
        c2, t2 = table.c[:, null_level, :2], table.t[:, null_level, :2]
        choice = same_sample_oracle(c2, t2, np.zeros(len(table.names), dtype=int))
        null_acc, null_time = float(pick(c2, choice).mean()), float(pick(t2, choice).sum())
        fixed_acc, fixed_time = float(c2.mean()), float(t2.sum() / 2)
        out["null"] = {"level": table.levels[null_level], "accuracy_gain": null_acc - fixed_acc,
                       "time_saved": 1 - null_time / fixed_time}
    return out


def report(title: str, table: Table, numbers: dict, n_boot: int) -> None:
    print()
    print("{} ({} problems; repeats {})".format(title, numbers["n"], ",".join(map(str, table.repeats))))
    print("  {:28} {:>9} {:>12} {:>14}".format("policy", "accuracy", "total time", "vs baseline"))
    base_acc, base_time = numbers["baseline"]

    def line(name: str, acc: float, total: float) -> None:
        print("  {:28} {:>8.1%} {:>11.0f}s {:>13.0%}".format(name, acc, total, 1 - total / base_time))

    for level, (acc, total) in numbers["fixed"].items():
        line("fixed " + level, acc, total)
    chosen = sorted(set(numbers["per_set_best"]))
    line("best fixed per set (" + ",".join(chosen) + ")", base_acc, base_time)
    line("oracle, same sample", *numbers["same_sample"])
    if "split_sample" in numbers:
        line("oracle, split samples", *numbers["split_sample"])
        c, t, bc, bt = numbers["_split"]
        samples = bootstrap(len(c), n_boot)
        saved = [1 - t[i].sum() / bt[i].sum() for i in samples]
        accuracy = [c[i].mean() - bc[i].mean() for i in samples]
        print("  split-sample oracle minus baseline, 95% interval: accuracy [{:+.1%}, {:+.1%}]  "
              "time saved [{:+.0%}, {:+.0%}]".format(*interval(accuracy), *interval(saved)))
        null = numbers["null"]
        same_saved = 1 - numbers["same_sample"][1] / base_time
        print("  same-level null ({}): accuracy {:+.1%}, time saved {:.0%}; the same-sample oracle "
              "clears it by {:+.0%} of time".format(null["level"], null["accuracy_gain"], null["time_saved"],
                                                     same_saved - null["time_saved"]))


def labels(runs: dict, names: list[str], levels: list[str]) -> list[dict]:
    out = []
    for name in names:
        entry = {"case": name, "set": set_of(name), "p_correct": {}, "seconds": {}, "n": {}}
        for level in levels:
            cells = [run for (case, lv, _), run in runs.items() if case == name and lv == level]
            if cells:
                entry["p_correct"][level] = float(np.mean([r["correct"] for r in cells]))
                entry["seconds"][level] = float(np.mean([r["seconds"] for r in cells]))
                entry["n"][level] = len(cells)
        out.append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, action="append", default=[])
    parser.add_argument("--import", dest="imports", action="append", default=[],
                        help="PATH=LEVEL[:REPEAT], the never-cut runs of a govern_live file")
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--levels", default="none,low,medium", help="the ladder, cheapest first")
    parser.add_argument("--repeats", default="0,1")
    parser.add_argument("--tolerance", type=float, default=0.01, help="accuracy given up for a cheaper fixed level")
    parser.add_argument("--null-level", default="medium")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--labels-out", type=Path)
    args = parser.parse_args()

    cases = {}
    for path in args.cases:
        cases.update({c["name"]: c for c in json.loads(path.read_text(encoding="utf-8"))})
    runs = load_runs(args.runs, args.imports, cases)
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    repeats = [int(r) for r in args.repeats.split(",")]
    names = complete_names(runs, sorted(cases), levels, repeats)
    if not names:
        print("no problem has a run at every level and repeat ({} runs read)".format(len(runs)))
        return 1
    null_level = levels.index(args.null_level) if args.null_level in levels else len(levels) - 1

    table = Table(runs, names, levels, repeats)
    sets = np.array([set_of(n) for n in names])
    report("All problems", table, ceiling(table, sets, args.tolerance, null_level), args.bootstrap)
    for name in sorted(set(sets)):
        rows = [n for n in names if set_of(n) == name]
        sub = Table(runs, rows, levels, repeats)
        report(name, sub, ceiling(sub, np.array([name] * len(rows)), args.tolerance, null_level), args.bootstrap)

    if args.labels_out:
        everything = sorted({case for case, _, _ in runs})
        args.labels_out.write_text(json.dumps(labels(runs, everything, levels), indent=1), encoding="utf-8")
        print("\nlabels for {} problems written to {}".format(len(everything), args.labels_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
