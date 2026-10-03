"""The arithmetic the go/no-go decisions rest on, checked on tables small enough to
work out by hand."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from effort_ceiling import Table, best_fixed, ceiling, labels  # noqa: E402

LEVELS = ["none", "low", "medium"]
SECONDS = {"none": 1.0, "low": 5.0, "medium": 10.0}

# Per problem and level, right or wrong on repeat 0 and repeat 1.
GRID = {
    "math-easy": {"none": (1, 1), "low": (1, 1), "medium": (1, 1)},
    "math-mid": {"none": (0, 0), "low": (1, 1), "medium": (1, 1)},
    "math-hard": {"none": (0, 0), "low": (0, 0), "medium": (1, 1)},
    "math-lucky": {"none": (1, 0), "low": (0, 0), "medium": (1, 1)},  # right once without thinking, by luck
}


def runs() -> dict:
    return {(name, level, repeat): {"correct": bool(cells[repeat]), "seconds": SECONDS[level]}
            for name, by_level in GRID.items() for level, cells in by_level.items() for repeat in (0, 1)}


@pytest.fixture
def table() -> Table:
    return Table(runs(), sorted(GRID), LEVELS, [0, 1])


def test_the_best_fixed_level_is_the_cheapest_close_enough_to_the_top(table) -> None:
    rows = np.ones(len(GRID), dtype=bool)
    assert LEVELS[best_fixed(table, rows, tolerance=0.01)] == "medium"
    assert LEVELS[best_fixed(table, rows, tolerance=0.6)] == "low"  # 50% is within 60 points of 100%


def test_the_oracles_and_the_null(table) -> None:
    numbers = ceiling(table, np.array(["math"] * len(GRID)), tolerance=0.01, null_level=2)
    assert numbers["baseline"] == (1.0, 40.0)
    # Same sample: the fastest right run, the lucky one included.
    assert numbers["same_sample"] == (1.0, 21.5)
    # Split samples: the lucky choice made on repeat 0 fails on repeat 1.
    assert numbers["split_sample"] == (0.875, 21.5)
    # Two repeats of medium are the same: noise alone gains nothing here.
    assert numbers["null"]["time_saved"] == 0.0 and numbers["null"]["accuracy_gain"] == 0.0


def test_labels_are_probabilities_per_level() -> None:
    entry = {e["case"]: e for e in labels(runs(), sorted(GRID), LEVELS)}["math-lucky"]
    assert entry["p_correct"] == {"none": 0.5, "low": 0.0, "medium": 1.0}
    assert entry["seconds"]["medium"] == 10.0 and entry["n"]["none"] == 2


def test_gate_a_eval_finds_a_planted_signal_and_writes_an_artifact(tmp_path, monkeypatch) -> None:
    import json

    import gate_a_eval
    from flyball.gate_a import GateAModel

    rng = np.random.default_rng(0)
    n = 200
    names = ["math-{:04d}".format(i) for i in range(n)]
    easy = rng.random(n) < 0.5  # easy problems are right even without thinking
    x = np.c_[easy + rng.normal(0, 0.1, n), rng.normal(0, 1, n)]
    labels = [{"case": name, "set": "math",
               "p_correct": {"none": float(e), "low": float(e), "medium": 1.0},
               "seconds": {"none": 1.0, "low": 5.0, "medium": 10.0}, "n": {}} for name, e in zip(names, easy)]
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    np.savez(tmp_path / "x.npz", features=x, names=np.array(names), latency=np.array([0.001]), spec=np.array("state"))
    monkeypatch.setattr(sys, "argv", [
        "gate_a_eval.py", "--labels", str(tmp_path / "labels.json"), "--features", "syn=" + str(tmp_path / "x.npz"),
        "--bootstrap", "50", "--artifact-out", str(tmp_path / "gate_a.json"), "--artifact-features", "syn",
        "--report-json", str(tmp_path / "report.json")])
    assert gate_a_eval.main() == 0

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    learned = report["logistic syn"]["all"]
    assert learned["accuracy"] >= 0.97 and learned["saved_vs_best"] > 0.3
    model = GateAModel.load(tmp_path / "gate_a.json")
    assert model.choose(model.predict([1.0, 0.0])) == "none"
    assert model.choose(model.predict([0.0, 0.0])) == "medium"


def test_a_schedule_probes_the_first_checkpoint_past_each_point() -> None:
    from compare_governors import mark_schedule

    from flyball.governor import GovernorConfig, probe_points

    trace = [{"position": p} for p in (150, 320, 480, 650, 1250, 2500)]
    mark_schedule(trace, probe_points(GovernorConfig(first_probe=300, growth=2.0, cap=None), 2500), "k")
    assert [r["position"] for r in trace if r["schedule"]["k"]] == [320, 650, 1250, 2500]
    mark_schedule(trace, probe_points(GovernorConfig(first_probe=1200, growth=2.0, cap=None), 2500), "late")
    assert [r["position"] for r in trace if r["schedule"]["late"]] == [1250, 2500]
