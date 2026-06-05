"""Tests for leakage-safe spatial neighbourhood features."""
from __future__ import annotations

import numpy as np

from src.spatial import (
    neighbour_features,
    oof_neighbour_features,
    radec_to_xyz,
)


def test_radec_to_xyz_on_unit_sphere():
    xyz = radec_to_xyz(np.array([0.0, 90.0, 180.0]), np.array([0.0, 0.0, 90.0]))
    assert xyz.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-12)
    # RA=0,Dec=0 -> +x ; RA=90,Dec=0 -> +y ; Dec=90 -> +z
    np.testing.assert_allclose(xyz[0], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(xyz[1], [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(xyz[2], [0, 0, 1], atol=1e-9)


def test_radec_wraparound_points_are_close():
    # RA 359.9 and 0.1 are ~0.2 deg apart, not ~360 deg
    xyz = radec_to_xyz(np.array([359.9, 0.1]), np.array([0.0, 0.0]))
    chord = np.linalg.norm(xyz[0] - xyz[1])
    assert chord < 0.01


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    xyz = radec_to_xyz(rng.uniform(0, 360, n), rng.uniform(-80, 80, n))
    y = rng.integers(0, 3, n)
    return xyz, y


def test_neighbour_features_shape_and_finite():
    xyz, y = _toy()
    ks = [5, 10]
    feats, names = neighbour_features(
        xyz, xyz, y, ks, n_classes=3, priors=np.array([0.6, 0.2, 0.2]),
        smoothing=1.0, max_k=20,
    )
    assert feats.shape == (len(xyz), len(names))
    assert np.isfinite(feats).all()
    # class-fraction columns are valid probabilities
    frac_cols = [i for i, nm in enumerate(names) if "frac" in nm]
    assert (feats[:, frac_cols] >= 0).all() and (feats[:, frac_cols] <= 1).all()


def test_oof_features_do_not_leak_own_label():
    # Flipping one row's OWN label must not change that row's OOF feature, because
    # its features are computed only from other folds.
    xyz, y = _toy(n=600, seed=1)
    fold_ids = np.arange(len(y)) % 5
    kw = {"ks": [5, 10], "n_classes": 3, "priors": np.array([1 / 3, 1 / 3, 1 / 3]),
          "smoothing": 1.0, "max_k": 20}
    base, _ = oof_neighbour_features(xyz, y, fold_ids, **kw)

    i = 7
    y2 = y.copy()
    y2[i] = (y2[i] + 1) % 3
    flipped, _ = oof_neighbour_features(xyz, y2, fold_ids, **kw)

    # row i's own feature is unchanged ...
    np.testing.assert_array_equal(base[i], flipped[i])
    # ... but the change is visible somewhere else (row i is a neighbour of others)
    assert not np.array_equal(base, flipped)
