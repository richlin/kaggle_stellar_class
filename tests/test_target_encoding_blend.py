"""Focused tests for the target-encoding blend candidate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_blend_module():
    module_path = Path("scripts/11_target_encoding_blend.py")
    spec = importlib.util.spec_from_file_location("target_encoding_blend_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_target_encoding_blend_uses_fixed_weights() -> None:
    blend_module = _load_blend_module()
    base = np.array([[1.0, 0.0, 0.0]])
    xgb = np.array([[0.0, 1.0, 0.0]])
    target_encoding = np.array([[0.0, 0.0, 1.0]])

    blended = blend_module.target_encoding_blend(base, xgb, target_encoding)

    np.testing.assert_allclose(blended, np.array([[0.5, 0.4, 0.1]]))


def test_make_target_encoding_blend_submission_preserves_id_order() -> None:
    blend_module = _load_blend_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [6, 5, 7], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = blend_module.make_target_encoding_blend_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [6, 5, 7]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
