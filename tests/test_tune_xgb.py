"""Focused tests for Phase 6 XGBoost tuning helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_tune_xgb_module():
    module_path = Path("scripts/05_tune_xgb.py")
    spec = importlib.util.spec_from_file_location("tune_xgb_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FixedTrial:
    def suggest_float(self, name, low, high, *, log=False):
        del log
        return (low + high) / 2

    def suggest_int(self, name, low, high):
        del name
        return (low + high) // 2


def test_sample_xgb_params_within_expected_bounds() -> None:
    tune_xgb = _load_tune_xgb_module()

    params = tune_xgb.sample_xgb_params(_FixedTrial())

    assert params["objective"] == "multi:softprob"
    assert params["num_class"] == 3
    assert 0.01 <= params["learning_rate"] <= 0.08
    assert 4 <= params["max_depth"] <= 10
    assert 1500 <= params["n_estimators"] <= 3000
    assert params["early_stopping_rounds"] == 50


def test_xgboost_frames_one_hot_encode_categoricals_consistently() -> None:
    tune_xgb = _load_tune_xgb_module()
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "cat": pd.Series(["a", "b"], dtype="category"),
        }
    )
    test = pd.DataFrame(
        {
            "num": [3.0],
            "cat": pd.Series(["b"], dtype="category"),
        }
    )

    train_encoded, test_encoded = tune_xgb.xgboost_frames(train, test)

    assert train_encoded.columns.tolist() == test_encoded.columns.tolist()
    assert {"cat_a", "cat_b"}.issubset(train_encoded.columns)


def test_objective_returns_finite_float_on_tiny_fixture() -> None:
    tune_xgb = _load_tune_xgb_module()
    X = pd.DataFrame(
        {
            "f1": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
            "f2": [0.2, 0.0, 1.2, 1.0, 2.2, 2.0],
        }
    )
    y = np.array([0, 0, 1, 1, 2, 2])
    params = {
        **tune_xgb.DEFAULT_XGB_PARAMS,
        "n_estimators": 5,
        "max_depth": 2,
        "early_stopping_rounds": 2,
        "n_jobs": 1,
    }

    score = tune_xgb.evaluate_xgb_params(X, y, params, n_splits=2, seed=7)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_make_tuned_xgb_submission_preserves_id_order() -> None:
    tune_xgb = _load_tune_xgb_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [8, 7, 9], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = tune_xgb.make_tuned_xgb_submission(sample_submission, probabilities, np.ones(3), encoder)

    assert submission["id"].tolist() == [8, 7, 9]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
