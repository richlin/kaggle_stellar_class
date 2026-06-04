"""Focused tests for the CV/threshold script helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import make_label_encoder


def _load_cv_module():
    module_path = Path("scripts/02_cv_threshold.py")
    spec = importlib.util.spec_from_file_location("cv_threshold_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_tuned_submission_preserves_ids_and_applies_multipliers() -> None:
    cv_threshold = _load_cv_module()
    encoder = make_label_encoder()
    probabilities = np.array([[0.40, 0.35, 0.25], [0.20, 0.45, 0.35]])
    multipliers = np.array([1.0, 0.7, 1.8])

    submission = cv_threshold.make_tuned_submission(
        pd.Series([577347, 577348]),
        probabilities,
        multipliers,
        encoder,
    )

    assert submission.columns.tolist() == ["id", "class"]
    assert submission["id"].tolist() == [577347, 577348]
    assert submission["class"].tolist() == ["STAR", "STAR"]


def test_recall_variation_flags_unstable_classes() -> None:
    cv_threshold = _load_cv_module()
    fold_recalls = [
        {"GALAXY": 0.95, "QSO": 0.96, "STAR": 0.94},
        {"GALAXY": 0.96, "QSO": 0.91, "STAR": 0.945},
    ]

    warnings = cv_threshold.recall_variation_warnings(fold_recalls, max_allowed_range=0.02)

    assert warnings == {"QSO": 0.04999999999999993}


def test_search_stable_multipliers_respects_class_regression_limit() -> None:
    cv_threshold = _load_cv_module()
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.46, 0.34, 0.20],
            [0.46, 0.34, 0.20],
            [0.48, 0.42, 0.10],
            [0.48, 0.42, 0.10],
            [0.51, 0.04, 0.45],
            [0.51, 0.04, 0.45],
        ]
    )
    fold_ids = np.array([1, 1, 1, 1, 1, 1])

    result = cv_threshold.search_stable_multipliers(
        y_true,
        probabilities,
        fold_ids,
        ["GALAXY", "QSO", "STAR"],
        grid=np.array([0.5, 0.75, 1.0, 1.25]),
        min_class_recall_delta=0.0,
        min_fold_score_delta=-1.0,
    )

    assert result["accepted"] is True
    assert result["score"] > result["baseline_score"]
    assert result["class_recall_deltas"]["GALAXY"] >= 0.0
