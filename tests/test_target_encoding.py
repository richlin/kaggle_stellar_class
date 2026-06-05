"""Focused tests for leakage-safe target encoding helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_target_encoding_module():
    module_path = Path("scripts/10_target_encoding.py")
    spec = importlib.util.spec_from_file_location("target_encoding_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_redshift_bins_returns_categorical_series() -> None:
    target_encoding = _load_target_encoding_module()
    bins = target_encoding.build_redshift_bins(pd.Series([0.01, 0.06, 0.2, 0.9, 2.0]))

    assert str(bins.dtype) == "category"
    assert bins.isna().sum() == 0


def test_fit_apply_target_encoding_uses_global_prior_for_unknown_categories() -> None:
    target_encoding = _load_target_encoding_module()
    train_categories = pd.Series(["a", "a", "b", "b"])
    y = np.array([0, 0, 1, 2])
    apply_categories = pd.Series(["a", "c"])

    encoded = target_encoding.fit_apply_target_encoding(
        train_categories,
        y,
        apply_categories,
        n_classes=3,
        smoothing=1.0,
    )

    assert encoded.shape == (2, 3)
    np.testing.assert_allclose(encoded.sum(axis=1), np.ones(2))
    assert encoded[0, 0] > encoded[0, 1]
    np.testing.assert_allclose(encoded[1], np.bincount(y, minlength=3) / len(y))


def test_make_target_encoding_submission_preserves_id_order() -> None:
    target_encoding = _load_target_encoding_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [4, 2, 3], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    submission = target_encoding.make_target_encoding_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [4, 2, 3]
    assert submission["class"].tolist() == ["GALAXY", "QSO", "STAR"]
