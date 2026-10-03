"""Calibration and small linear heads, owned by flyball rather than by any backend.

A probability from a decision model is only an opinion: Laya's are overconfident,
a kNN score is not a probability at all, and a probe on the model's hidden states
is whatever its training made it. flyball calibrates every score it acts on, on
held-out data, so that a threshold means the same thing whichever backend produced
the score, and backends can be compared on equal terms.

Applying a head or a calibration is plain Python, so the proxy needs nothing
heavier than the standard library. Fitting needs numpy (`pip install
flyballd[research]`), imported only by the fitting functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


@dataclass
class Platt:
    """p = sigmoid(a * score + b), with the score on the logit scale."""

    a: float = 1.0
    b: float = 0.0

    def apply(self, score: float) -> float:
        return sigmoid(self.a * score + self.b)

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict) -> "Platt":
        return cls(float(data.get("a", 1.0)), float(data.get("b", 0.0)))


@dataclass
class LogisticHead:
    """A standardised linear model: logit = w . (x - mean) / scale + bias."""

    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: float = 0.0

    def logit(self, x: Sequence[float]) -> float:
        if len(x) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} features, got {len(x)}")
        return self.bias + sum(w * (v - m) / s for w, v, m, s in zip(self.weights, x, self.mean, self.scale))

    def predict(self, x: Sequence[float]) -> float:
        return sigmoid(self.logit(x))

    def as_dict(self) -> dict:
        return {"mean": self.mean, "scale": self.scale, "weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticHead":
        return cls([float(v) for v in data["mean"]], [float(v) for v in data["scale"]],
                   [float(v) for v in data["weights"]], float(data.get("bias", 0.0)))


def fit_head(x, y, l2: float = 1e-2, steps: int = 400, lr: float = 0.1) -> LogisticHead:
    """Gradient descent on the standardised features; `l2` is chosen by the caller
    on held-out folds."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = (x - mean) / scale
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(z @ w + b, -30, 30)))
        w -= lr * (z.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return LogisticHead(mean.tolist(), scale.tolist(), w.tolist(), float(b))


def head_logits(head: LogisticHead, x):
    """The head's logits for a whole matrix at once (numpy)."""
    import numpy as np

    z = (np.asarray(x, dtype=np.float64) - np.asarray(head.mean)) / np.asarray(head.scale)
    return z @ np.asarray(head.weights) + head.bias


def fit_platt(scores, labels, steps: int = 2000, lr: float = 0.05) -> Platt:
    """Fit on held-out scores (logits), never on the data the scorer was trained on."""
    import numpy as np

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(a * s + b, -30, 30)))
        a -= lr * float(np.mean((p - y) * s))
        b -= lr * float(np.mean(p - y))
    return Platt(a, b)
