"""Focused tests for the class-adjusted LightGBM candidate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder


def _load_class_weight_module():
    module_path = Path("scripts/13_class_weight_lgbm.py")
    spec = importlib.util.spec_from_file_location("class_weight_lgbm_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_class_adjusted_sample_weights_are_positive_and_normalized() -> None:
    class_weight_lgbm = _load_class_weight_module()
    y = np.array([0, 0, 0, 1, 2])
    class_adjustment = np.array([1.0, 1.1, 1.05])

    weights = class_weight_lgbm.class_adjusted_sample_weights(y, class_adjustment)

    assert weights.shape == y.shape
    assert np.all(weights > 0)
    assert weights[y == 1][0] > weights[y == 0][0]
    assert weights.mean() == pytest.approx(1.0)


def test_make_class_weight_submission_preserves_id_order() -> None:
    class_weight_lgbm = _load_class_weight_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [12, 10, 11], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    submission = class_weight_lgbm.make_class_weight_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [12, 10, 11]
    assert submission["class"].tolist() == ["GALAXY", "QSO", "STAR"]
