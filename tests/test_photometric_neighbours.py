"""Tests for photometric-neighbour feature construction (scripts/28_photometric_neighbours)."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.spatial import neighbour_features, oof_neighbour_features


def _toy(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    color = rng.standard_normal((n, 4))  # u_g, g_r, r_i, i_z
    mags = rng.standard_normal((n, 5))   # u, g, r, i, z
    xyz = rng.standard_normal((n, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    y = rng.integers(0, 3, n)
    return color, mags, xyz, y


def test_color_space_features_shape_and_finite():
    color, _, _, y = _toy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(color)
    feats, names = neighbour_features(
        scaled, scaled, y, [5, 10],
        n_classes=3, priors=np.array([0.6, 0.2, 0.2]),
        smoothing=1.0, max_k=20,
    )
    assert feats.shape == (len(color), len(names))
    assert np.isfinite(feats).all()
    frac_cols = [i for i, nm in enumerate(names) if "frac" in nm]
    assert (feats[:, frac_cols] >= 0).all() and (feats[:, frac_cols] <= 1).all()


def test_magnitude_space_features_shape_and_finite():
    _, mags, _, y = _toy(n=300, seed=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(mags)
    feats, names = neighbour_features(
        scaled, scaled, y, [5, 10],
        n_classes=3, priors=np.array([0.6, 0.2, 0.2]),
        smoothing=1.0, max_k=20,
    )
    assert feats.shape == (300, len(names))
    assert np.isfinite(feats).all()


def test_joint_sphere_color_features_shape_and_finite():
    color, _, xyz, y = _toy(n=200, seed=2)
    joint = np.hstack([xyz, color])  # 7D
    scaler = StandardScaler()
    scaled = scaler.fit_transform(joint)
    feats, names = neighbour_features(
        scaled, scaled, y, [5],
        n_classes=3, priors=np.array([1 / 3, 1 / 3, 1 / 3]),
        smoothing=1.0, max_k=10,
    )
    assert feats.shape == (200, len(names))
    assert np.isfinite(feats).all()


def test_oof_color_features_do_not_leak_own_label():
    """Flipping a row's own label must not change its OOF colour-space features."""
    color, _, _, y = _toy(n=500, seed=7)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(color)
    fold_ids = np.arange(500) % 5
    kw = {
        "ks": [5], "n_classes": 3,
        "priors": np.array([1 / 3, 1 / 3, 1 / 3]),
        "smoothing": 1.0, "max_k": 10,
    }
    base, _ = oof_neighbour_features(scaled, y, fold_ids, **kw)
    i = 13
    y2 = y.copy()
    y2[i] = (y2[i] + 1) % 3
    flipped, _ = oof_neighbour_features(scaled, y2, fold_ids, **kw)
    np.testing.assert_array_equal(base[i], flipped[i])
    assert not np.array_equal(base, flipped)
