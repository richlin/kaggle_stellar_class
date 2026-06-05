"""Focused tests for the boundary-v1 experiment script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_boundary_module():
    module_path = Path("scripts/05_boundary_features.py")
    spec = importlib.util.spec_from_file_location("boundary_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_boundary_candidate_requires_oof_gain() -> None:
    boundary = _load_boundary_module()

    rejected = boundary.select_boundary_candidate(
        candidate_score=0.9658,
        reference_score=0.965925,
    )
    accepted = boundary.select_boundary_candidate(
        candidate_score=0.9662,
        reference_score=0.965925,
    )

    assert rejected["accepted"] is False
    assert accepted["accepted"] is True


def test_make_boundary_submission_preserves_id_order() -> None:
    boundary = _load_boundary_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [3, 1, 2], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = boundary.make_boundary_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [3, 1, 2]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
