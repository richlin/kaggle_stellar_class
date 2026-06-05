"""Focused tests for high-confidence pseudo-label helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_pseudolabel_module():
    module_path = Path("scripts/08_pseudolabel.py")
    spec = importlib.util.spec_from_file_location("pseudolabel_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_pseudolabels_requires_high_confidence_and_margin() -> None:
    pseudolabel = _load_pseudolabel_module()
    probabilities = np.array(
        [
            [0.996, 0.002, 0.002],
            [0.800, 0.190, 0.010],
            [0.100, 0.450, 0.450],
            [0.001, 0.001, 0.998],
        ]
    )

    selected = pseudolabel.select_pseudolabels(probabilities, min_probability=0.995, min_margin=0.75)

    assert selected["mask"].tolist() == [True, False, False, True]
    assert selected["labels"].tolist() == [0, 2]


def test_make_pseudolabel_submission_preserves_id_order() -> None:
    pseudolabel = _load_pseudolabel_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [5, 4, 6], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = pseudolabel.make_pseudolabel_submission(sample_submission, probabilities, encoder)

    assert submission["id"].tolist() == [5, 4, 6]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
