"""Focused tests for Phase 5 ensemble helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.validation import search_class_multipliers


def _load_ensemble_module():
    module_path = Path("scripts/04_ensemble.py")
    spec = importlib.util.spec_from_file_location("ensemble_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weighted_blend_preserves_shape_and_normalizes_weights() -> None:
    ensemble = _load_ensemble_module()
    first = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5]])
    second = np.array([[0.3, 0.7, 0.0], [0.8, 0.1, 0.1]])

    blended = ensemble.weighted_probability_blend([first, second], np.array([2.0, 1.0]))

    expected = (first * (2.0 / 3.0)) + (second * (1.0 / 3.0))
    np.testing.assert_allclose(blended, expected)
    assert blended.shape == first.shape


def test_search_blend_weights_prefers_better_oof_blend() -> None:
    ensemble = _load_ensemble_module()
    y_true = np.array([0, 0, 1, 1, 2, 2])
    model_a = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.42, 0.45, 0.13],
            [0.30, 0.55, 0.15],
            [0.32, 0.53, 0.15],
            [0.28, 0.42, 0.30],
            [0.25, 0.37, 0.38],
        ]
    )
    model_b = np.array(
        [
            [0.65, 0.25, 0.10],
            [0.70, 0.20, 0.10],
            [0.20, 0.70, 0.10],
            [0.15, 0.75, 0.10],
            [0.10, 0.25, 0.65],
            [0.10, 0.20, 0.70],
        ]
    )
    fold_ids = np.array([1, 2, 1, 2, 1, 2])

    result = ensemble.search_blend_weights(
        y_true,
        [model_a, model_b],
        fold_ids,
        ["GALAXY", "QSO", "STAR"],
    )

    assert result["score"] == 1.0
    assert result["weights"][1] > result["weights"][0]
    assert result["per_class_recall"] == {"GALAXY": 1.0, "QSO": 1.0, "STAR": 1.0}


def test_make_ensemble_submission_preserves_id_order() -> None:
    ensemble = _load_ensemble_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [20, 10, 30], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = ensemble.make_ensemble_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [20, 10, 30]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]


def test_continuous_threshold_search_beats_or_matches_coarse_grid() -> None:
    ensemble = _load_ensemble_module()
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.42, 0.45, 0.13],
            [0.20, 0.70, 0.10],
            [0.15, 0.75, 0.10],
            [0.10, 0.25, 0.65],
            [0.10, 0.20, 0.70],
        ]
    )
    fold_ids = np.array([1, 2, 1, 2, 1, 2])
    coarse_multipliers, coarse_score = search_class_multipliers(y_true, probabilities)

    result = ensemble.search_continuous_multipliers(
        y_true,
        probabilities,
        fold_ids,
        ["GALAXY", "QSO", "STAR"],
        initial_multipliers=coarse_multipliers,
    )

    assert result["score"] >= coarse_score
    assert result["accepted"] is True
    assert result["multipliers"].shape == (3,)
