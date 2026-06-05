"""Tests for external-labelled spatial reference features."""
from __future__ import annotations

import numpy as np

from src.external_spatial import (
    external_reference_oof_features,
    make_append_sample_weights,
)
from src.spatial import radec_to_xyz


def test_external_reference_oof_features_do_not_use_validation_labels() -> None:
    comp_xyz = radec_to_xyz(
        np.array([0.0, 0.1, 100.0, 100.1, 200.0, 200.1]),
        np.zeros(6),
    )
    comp_y = np.array([0, 0, 1, 1, 2, 2])
    ext_xyz = radec_to_xyz(np.array([0.05, 100.05, 200.05]), np.zeros(3))
    ext_y = np.array([0, 1, 2])
    fold_ids = np.array([0, 0, 1, 1, 2, 2])

    base, _ = external_reference_oof_features(
        comp_xyz,
        comp_y,
        fold_ids,
        ext_xyz,
        ext_y,
        ks=[1],
        n_classes=3,
        priors=np.array([1 / 3, 1 / 3, 1 / 3]),
        smoothing=0.0,
        max_k=1,
    )

    flipped_y = comp_y.copy()
    flipped_y[0] = 2
    flipped, _ = external_reference_oof_features(
        comp_xyz,
        flipped_y,
        fold_ids,
        ext_xyz,
        ext_y,
        ks=[1],
        n_classes=3,
        priors=np.array([1 / 3, 1 / 3, 1 / 3]),
        smoothing=0.0,
        max_k=1,
    )

    np.testing.assert_array_equal(base[0], flipped[0])


def test_external_reference_labels_can_influence_validation_features() -> None:
    comp_xyz = radec_to_xyz(np.array([0.0, 100.0, 200.0]), np.zeros(3))
    comp_y = np.array([0, 1, 2])
    ext_xyz = radec_to_xyz(np.array([0.01]), np.zeros(1))
    fold_ids = np.array([0, 1, 2])

    ext_zero, _ = external_reference_oof_features(
        comp_xyz,
        comp_y,
        fold_ids,
        ext_xyz,
        np.array([0]),
        ks=[1],
        n_classes=3,
        priors=np.array([1 / 3, 1 / 3, 1 / 3]),
        smoothing=0.0,
        max_k=1,
    )
    ext_two, _ = external_reference_oof_features(
        comp_xyz,
        comp_y,
        fold_ids,
        ext_xyz,
        np.array([2]),
        ks=[1],
        n_classes=3,
        priors=np.array([1 / 3, 1 / 3, 1 / 3]),
        smoothing=0.0,
        max_k=1,
    )

    assert not np.array_equal(ext_zero[0], ext_two[0])


def test_make_append_sample_weights_only_downweights_external_rows() -> None:
    weights = make_append_sample_weights(n_competition=3, n_external=2, external_weight=0.25)

    np.testing.assert_allclose(weights, np.array([1.0, 1.0, 1.0, 0.25, 0.25]))
