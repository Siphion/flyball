"""Gate B, the mid-stream governor: has the reasoning done its job yet?

This module is the runtime half. Given the user's request and the reasoning streamed
so far, it returns a calibrated probability that closing the reasoning now would
still produce the right answer. Training lives in tools/train_gate_b.py.

The first backend is deliberately small: a logistic regression over cheap features
plus a hashed bag of words from the end of the reasoning. It runs in microseconds on
a CPU, needs nothing beyond the standard library and NumPy, and is stored as plain
JSON rather than a pickle, so a model file can be read, diffed and trusted. If it
cannot beat a fixed reasoning budget, a larger backend has to justify itself with
numbers; if it can, it is the baseline every larger backend has to beat.

Numbers in the text are replaced by a placeholder before hashing. The model must
learn what a settled answer *looks like*, not memorise which numbers were answers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_VERSION = 2
HASH_DIMS = 4096
TAIL_CHARS = 300
WINDOW_CHARS = 300

CONCLUSION = re.compile(
    r"(?i)\b(so|thus|therefore|hence)\b[^.\n]{0,80}\b(answer|result|total|is|=|equals)\b"
    r"|\bfinal answer\b|\bthe answer is\b|\bboxed\b"
)
VERIFICATION = re.compile(
    r"(?i)\b(let me|let's|let us)\s+(double.?check|verify|check|confirm|re-?check)\b"
    r"|\bwait\b|\bactually\b|\bhmm\b|\balternatively\b|\bon second thought\b"
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
WORD = re.compile(r"[a-z]+|<num>|[=+\-*/]")


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in NUMBER.findall(text)]


def _hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % HASH_DIMS


def dense_features(prompt: str, prefix: str) -> dict[str, float]:
    """Hand-made features. Each one is something a governor can know mid-stream."""
    window = prefix[-WINDOW_CHARS:]
    earlier = prefix[:-WINDOW_CHARS]

    conclusions = list(CONCLUSION.finditer(prefix))
    since_conclusion = len(prefix) - conclusions[-1].end() if conclusions else len(prefix)
    after_last = prefix[conclusions[-1].end():] if conclusions else ""

    prefix_numbers = _numbers(prefix)
    prompt_numbers = set(_numbers(prompt))
    window_numbers = set(_numbers(window))
    earlier_numbers = set(_numbers(earlier))

    # The number that follows each conclusion phrase: if the same one keeps coming
    # back, the reasoning has probably settled on it.
    concluded_values = []
    for match in conclusions:
        follow = _numbers(prefix[match.end(): match.end() + 40])
        if follow:
            concluded_values.append(follow[0])
    repeats = max((concluded_values.count(v) for v in set(concluded_values)), default=0)

    return {
        "log_position": math.log1p(len(prefix)),
        "conclusions": float(len(conclusions)),
        "has_conclusion": 1.0 if conclusions else 0.0,
        "log_since_conclusion": math.log1p(since_conclusion),
        "verifications": float(len(VERIFICATION.findall(prefix))),
        "verifying_after_conclusion": 1.0 if VERIFICATION.search(after_last) else 0.0,
        "repeated_concluded_value": float(repeats),
        "distinct_numbers": math.log1p(len(set(prefix_numbers))),
        "new_numbers_in_window": float(len(window_numbers - earlier_numbers)),
        "prompt_numbers_used": (
            len(prompt_numbers & set(prefix_numbers)) / len(prompt_numbers) if prompt_numbers else 1.0
        ),
        # How hard the problem looks. Measured on GSM8K, risk tracks difficulty more
        # than position: long reasoning belongs to hard problems, and that is where a
        # cut is most likely to be premature.
        "log_prompt_chars": math.log1p(len(prompt)),
        "prompt_numbers": float(len(prompt_numbers)),
    }


def hashed_tail(prefix: str) -> dict[int, float]:
    """Unigrams and bigrams of the end of the reasoning, numbers masked."""
    tail = NUMBER.sub(" <num> ", prefix[-TAIL_CHARS:].lower())
    words = WORD.findall(tail)
    counts: dict[int, float] = {}
    for token in words + [a + "_" + b for a, b in zip(words, words[1:])]:
        index = _hash(token)
        counts[index] = counts.get(index, 0.0) + 1.0
    return counts


DENSE_NAMES = list(dense_features("", "").keys())


def vectorise(prompt: str, prefix: str) -> np.ndarray:
    dense = dense_features(prompt, prefix)
    vector = np.zeros(len(DENSE_NAMES) + HASH_DIMS, dtype=np.float64)
    vector[: len(DENSE_NAMES)] = [dense[name] for name in DENSE_NAMES]
    sparse = hashed_tail(prefix)
    if sparse:
        norm = math.sqrt(sum(v * v for v in sparse.values()))
        for index, value in sparse.items():
            vector[len(DENSE_NAMES) + index] = value / norm
    return vector


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


@dataclass
class GateBModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    # Platt scaling fitted on a held-out split: the raw score is not a probability.
    calib_a: float = 1.0
    calib_b: float = 0.0

    def raw_score(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - self.mean) / self.scale) @ self.weights + self.bias

    def probability(self, matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(self.calib_a * self.raw_score(matrix) + self.calib_b)

    def prob_done(self, prompt: str, prefix: str) -> float:
        """Calibrated probability that closing the reasoning now is safe."""
        return float(self.probability(vectorise(prompt, prefix)[None, :])[0])

    def save(self, path: Path) -> None:
        payload = {
            "kind": "flyball.gate_b.logistic",
            "feature_version": FEATURE_VERSION,
            "hash_dims": HASH_DIMS,
            "dense_names": DENSE_NAMES,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "calib_a": self.calib_a,
            "calib_b": self.calib_b,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GateBModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("feature_version") != FEATURE_VERSION or payload.get("hash_dims") != HASH_DIMS:
            raise ValueError("model was trained with different features; retrain it")
        return cls(
            weights=np.array(payload["weights"]),
            bias=payload["bias"],
            mean=np.array(payload["mean"]),
            scale=np.array(payload["scale"]),
            calib_a=payload["calib_a"],
            calib_b=payload["calib_b"],
        )


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 3000) -> GateBModel:
    """Plain gradient descent with Adam: small data, no need for anything heavier."""
    # Only the hand-made features are standardised. The hashed word counts are
    # already normalised and almost always zero: standardising them would turn a word
    # seen once into an enormous value, and the model would learn that noise.
    dense = len(DENSE_NAMES)
    mean = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    mean[:dense] = x[:, :dense].mean(axis=0)
    scale[:dense] = x[:, :dense].std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    n, d = z.shape
    w = np.zeros(d)
    b = 0.0
    m_w = np.zeros(d); v_w = np.zeros(d); m_b = 0.0; v_b = 0.0
    lr, beta1, beta2, eps = 0.05, 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        p = _sigmoid(z @ w + b)
        grad_w = z.T @ (p - y) / n + l2 * w / n
        grad_b = float(np.mean(p - y))
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w**2
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b**2
        w -= lr * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
        b -= lr * (m_b / (1 - beta1**t)) / (math.sqrt(v_b / (1 - beta2**t)) + eps)
    return GateBModel(weights=w, bias=b, mean=mean, scale=scale)


def fit_platt(model: GateBModel, x: np.ndarray, y: np.ndarray, steps: int = 2000) -> None:
    """Fit the calibration on held-out data, never on the training split."""
    s = model.raw_score(x)
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = _sigmoid(a * s + b)
        a -= 0.05 * float(np.mean((p - y) * s))
        b -= 0.05 * float(np.mean(p - y))
    model.calib_a, model.calib_b = a, b
