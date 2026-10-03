"""Shared evaluation helpers for the research tools.

One definition of each measure, so that every tool splits, scores and bootstraps
the same way:

- `fold_of`: problems, never single checkpoints or runs, are split into folds, by
  a hash of the problem's name, so a fold never mixes two runs of one problem;
- `auc` and `ece`: how well a score ranks, and how well it is calibrated;
- `bootstrap`: resampling over problems, for intervals;
- `set_of`: the benchmark a case comes from, read from its name.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

FOLDS = 5
SETS = ("mmlu-pro", "math", "gsm8k", "mbpp")


def fold_of(case: str, folds: int = FOLDS) -> int:
    return int(hashlib.sha256(case.encode()).hexdigest(), 16) % folds


def set_of(case: str) -> str:
    return next((name for name in SETS if case.startswith(name + "-")), case.split("-")[0])


def auc(scores, labels) -> float:
    """Area under the ROC curve, with tied scores sharing their average rank."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ece(prob, labels, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    prob = np.asarray(prob, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & ((prob < high) if high < 1 else (prob <= high))
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - labels[mask].mean())
    return float(total)


def bootstrap(n: int, n_boot: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """Resampled index sets over `n` problems."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def interval(values, low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def fit_logistic(x, y, l2: float = 1e-2, steps: int = 400, lr: float = 0.1):
    """A standardised logistic regression (calibration.LogisticHead)."""
    from flyball.calibration import fit_head

    return fit_head(x, y, l2=l2, steps=steps, lr=lr)


def predict(head, x) -> np.ndarray:
    from flyball.calibration import head_logits

    return 1 / (1 + np.exp(-np.clip(head_logits(head, x), -30, 30)))
