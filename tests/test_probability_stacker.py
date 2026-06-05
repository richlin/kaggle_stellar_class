"""Focused tests for saved-probability stacking helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_stacker_module():
    module_path = Path("scripts/07_probability_stacker.py")
    spec = importlib.util.spec_from_file_location("stacker_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_stack_features_concatenates_probabilities_and_margins() -> None:
    stacker = _load_stacker_module()
    first = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]])
    second = np.array([[0.4, 0.5, 0.1], [0.3, 0.3, 0.4]])

    features = stacker.build_stack_features([first, second])

    assert features.shape == (2, 10)
    np.testing.assert_allclose(features[:, :3], first)
    np.testing.assert_allclose(features[:, 3:6], second)
    np.testing.assert_allclose(features[:, 6], [0.5, 0.4])
    np.testing.assert_allclose(features[:, 8], [0.1, 0.1])


def test_select_stacker_candidate_requires_reference_gain() -> None:
    stacker = _load_stacker_module()

    rejected = stacker.select_stacker_candidate(0.965, 0.966)
    accepted = stacker.select_stacker_candidate(0.967, 0.966)

    assert rejected["accepted"] is False
    assert accepted["accepted"] is True


def test_make_stacker_submission_preserves_id_order() -> None:
    stacker = _load_stacker_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [2, 1, 3], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = stacker.make_stacker_submission(sample_submission, probabilities, np.ones(3), encoder)

    assert submission["id"].tolist() == [2, 1, 3]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
