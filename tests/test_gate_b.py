"""Gate B model: deterministic features, an honest fit, a trustworthy file format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flyball.gate_b import (
    DENSE_NAMES,
    GateBModel,
    dense_features,
    fit_logistic,
    fit_platt,
    hashed_tail,
    vectorise,
)


def test_features_are_deterministic() -> None:
    a = vectorise("A pen costs 3.", "So the total is 13. Let me verify.")
    b = vectorise("A pen costs 3.", "So the total is 13. Let me verify.")
    assert np.array_equal(a, b)


def test_numbers_are_masked_before_hashing() -> None:
    # The model must learn what a settled answer looks like, not which numbers
    # happened to be answers in the training set.
    assert hashed_tail("so the total is 13") == hashed_tail("so the total is 9041")


def test_settling_on_one_value_is_visible() -> None:
    settled = dense_features("", "So the answer is 13. Checking again, so the answer is 13.")
    wavering = dense_features("", "So the answer is 13. Hmm, so the answer is 14.")
    assert settled["repeated_concluded_value"] > wavering["repeated_concluded_value"]


def test_prompt_numbers_used_counts_the_givens() -> None:
    assert dense_features("Add 3 and 7.", "3 plus 7 is 10")["prompt_numbers_used"] == 1.0
    assert dense_features("Add 3 and 7.", "Let me think")["prompt_numbers_used"] == 0.0


def _synthetic(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(DENSE_NAMES) + 8))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(float)
    return x, y


def test_fit_separates_a_learnable_signal() -> None:
    x, y = _synthetic()
    model = fit_logistic(x[:300], y[:300], l2=1.0, steps=1500)
    accuracy = np.mean((model.probability(x[300:]) >= 0.5) == (y[300:] == 1))
    assert accuracy > 0.9


def test_calibration_is_fitted_on_held_out_data() -> None:
    x, y = _synthetic(seed=1)
    model = fit_logistic(x[:200], y[:200], l2=1.0, steps=1500)
    before = (model.calib_a, model.calib_b)
    fit_platt(model, x[200:], y[200:])
    assert (model.calib_a, model.calib_b) != before


def test_model_round_trips_through_plain_json(tmp_path) -> None:
    x, y = _synthetic(n=50)
    width = len(vectorise("", ""))
    x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
    model = fit_logistic(x, y, steps=50)
    path = tmp_path / "model.json"
    model.save(path)
    assert json.loads(path.read_text())["kind"] == "flyball.gate_b.logistic"
    loaded = GateBModel.load(path)
    assert loaded.prob_done("q", "so the answer is 4") == pytest.approx(
        model.prob_done("q", "so the answer is 4")
    )


def test_a_model_with_other_features_is_refused(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"feature_version": 0, "hash_dims": 1}))
    with pytest.raises(ValueError):
        GateBModel.load(path)
