"""Focused tests for the STAR-safe blend candidate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_star_safe_module():
    module_path = Path("scripts/06_star_safe_blend.py")
    spec = importlib.util.spec_from_file_location("star_safe_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_star_safe_blend_uses_fixed_normalized_weights() -> None:
    star_safe = _load_star_safe_module()
    first = np.array([[1.0, 0.0, 0.0]])
    second = np.array([[0.0, 1.0, 0.0]])

    blended = star_safe.star_safe_blend(first, second)

    np.testing.assert_allclose(blended, np.array([[0.6, 0.4, 0.0]]))


def test_make_star_safe_submission_preserves_id_order() -> None:
    star_safe = _load_star_safe_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [9, 7, 8], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    submission = star_safe.make_star_safe_submission(sample_submission, probabilities, encoder)

    assert submission["id"].tolist() == [9, 7, 8]
    assert submission["class"].tolist() == ["GALAXY", "QSO", "STAR"]
