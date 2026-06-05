"""Tests for Task 24 transductive spatial helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.transductive_spatial import (
    build_probability_meta_features,
    full_train_cluster_class_rates,
    oof_cluster_class_rates,
    weighted_graph_probabilities,
)


def test_weighted_graph_probabilities_excludes_self_reference() -> None:
    query = np.array([[0.0, 0.0, 1.0]], dtype=float)
    ref = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    ref_probabilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    graph_probabilities = weighted_graph_probabilities(
        query,
        ref,
        ref_probabilities,
        n_neighbors=2,
        self_reference_indices=np.array([0]),
    )

    assert graph_probabilities.shape == (1, 3)
    assert graph_probabilities[0, 0] == 0.0
    np.testing.assert_allclose(graph_probabilities.sum(axis=1), 1.0)


def test_oof_cluster_class_rates_do_not_leak_own_label() -> None:
    cluster_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    fold_ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    priors = np.array([1 / 3, 1 / 3, 1 / 3])

    base = oof_cluster_class_rates(
        cluster_ids,
        y,
        fold_ids,
        n_clusters=3,
        n_classes=3,
        priors=priors,
        smoothing=1.0,
    )
    y_flipped = y.copy()
    y_flipped[0] = 2
    flipped = oof_cluster_class_rates(
        cluster_ids,
        y_flipped,
        fold_ids,
        n_clusters=3,
        n_classes=3,
        priors=priors,
        smoothing=1.0,
    )

    np.testing.assert_array_equal(base[0], flipped[0])
    assert not np.array_equal(base, flipped)


def test_full_train_cluster_class_rates_are_smoothed_probabilities() -> None:
    train_cluster_ids = np.array([0, 0, 1, 1])
    test_cluster_ids = np.array([0, 1, 2])
    y = np.array([0, 0, 1, 2])
    priors = np.array([0.5, 0.25, 0.25])

    rates = full_train_cluster_class_rates(
        train_cluster_ids,
        test_cluster_ids,
        y,
        n_clusters=3,
        n_classes=3,
        priors=priors,
        smoothing=2.0,
    )

    assert rates.shape == (3, 3)
    assert np.isfinite(rates).all()
    np.testing.assert_allclose(rates.sum(axis=1), 1.0)
    np.testing.assert_allclose(rates[2], priors)


def test_probability_meta_features_have_matching_finite_columns() -> None:
    train_probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    test_probabilities = np.array(
        [
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
        ]
    )

    train_meta, test_meta = build_probability_meta_features(
        {"base": train_probabilities},
        {"base": test_probabilities},
    )

    assert train_meta.columns.tolist() == test_meta.columns.tolist()
    assert train_meta.shape[0] == 3
    assert test_meta.shape[0] == 2
    assert np.isfinite(train_meta.to_numpy()).all()
    assert {"base_p0", "base_margin", "base_entropy"}.issubset(train_meta.columns)


def test_make_submission_preserves_id_order() -> None:
    module_path = Path("scripts/17_transductive_spatial.py")
    spec = importlib.util.spec_from_file_location("transductive_spatial_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [3, 1, 2], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
        ]
    )

    submission = module.make_submission(sample_submission, probabilities, np.ones(3), encoder)

    assert submission["id"].tolist() == [3, 1, 2]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]


def test_apply_galaxy_overrides_only_flips_targeted_rows() -> None:
    module_path = Path("scripts/18_galaxy_residual.py")
    spec = importlib.util.spec_from_file_location("galaxy_residual_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    incumbent_pred = np.array([2, 2, 1, 1, 0])
    star_to_galaxy = np.array([0.7, 0.2, 0.9, 0.1, 0.9])
    qso_to_galaxy = np.array([0.1, 0.8, 0.6, 0.2, 0.9])

    corrected = module.apply_galaxy_overrides(
        incumbent_pred,
        star_to_galaxy,
        qso_to_galaxy,
        star_threshold=0.5,
        qso_threshold=0.5,
    )

    assert corrected.tolist() == [0, 2, 0, 1, 0]


def test_loo_xgb_submission_preserves_id_order() -> None:
    module_path = Path("scripts/25_loo_spatial_xgb_final.py")
    spec = importlib.util.spec_from_file_location("loo_xgb_final_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [11, 10, 12], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    submission = module.make_submission(sample_submission, probabilities, np.ones(3), encoder)

    assert submission["id"].tolist() == [11, 10, 12]
    assert submission["class"].tolist() == ["STAR", "GALAXY", "QSO"]
