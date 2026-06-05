"""Focused tests for the multi-model blend candidate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder


def _load_multi_blend_module():
    module_path = Path("scripts/12_multi_blend.py")
    spec = importlib.util.spec_from_file_location("multi_blend_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multi_model_blend_uses_fixed_normalized_weights() -> None:
    multi_blend = _load_multi_blend_module()
    probabilities = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.2, 0.3, 0.5]]),
    ]

    blended = multi_blend.multi_model_blend(probabilities)

    expected = (
        (probabilities[0] * 0.23)
        + (probabilities[1] * 0.44)
        + (probabilities[2] * 0.28)
        + (probabilities[3] * 0.05)
    )
    np.testing.assert_allclose(blended, expected)


def test_multi_model_blend_rejects_wrong_number_of_arrays() -> None:
    multi_blend = _load_multi_blend_module()

    with pytest.raises(ValueError, match="exactly 4 probability arrays"):
        multi_blend.multi_model_blend([np.array([[1.0, 0.0, 0.0]])])


def test_make_multi_blend_submission_preserves_id_order() -> None:
    multi_blend = _load_multi_blend_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [3, 1, 2], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = multi_blend.make_multi_blend_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [3, 1, 2]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
