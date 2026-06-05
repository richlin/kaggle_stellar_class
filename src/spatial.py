"""Leakage-safe spatial neighbourhood features on sky position.

The dataset's class is clustered in sky position (position-only LightGBM reaches
~0.68 balanced accuracy; 10 nearest objects share a class ~68% of the time vs
~49% chance). A gradient-boosted model on raw ``alpha``/``delta`` can only make
axis-aligned splits, so it cannot represent "what fraction of my spatial
neighbours are class c". These helpers build that signal as out-of-fold k-NN
class-fraction features.

These features depend on the target, so they live here (fold-aware) rather than
in the pure, stateless ``src.features.build_feature_frame``.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-9


def radec_to_xyz(alpha_deg: np.ndarray, delta_deg: np.ndarray) -> np.ndarray:
    """Map (RA, Dec) in degrees to 3D unit-sphere coordinates.

    Using the unit sphere makes neighbour distances great-circle-consistent and
    removes the RA wraparound discontinuity at 0/360 degrees.
    """
    a = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    d = np.deg2rad(np.asarray(delta_deg, dtype=float))
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], axis=1)


def _features_from_neighbours(
    dist: np.ndarray,
    labels: np.ndarray,
    ks: list[int],
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, list[str]]:
    """Build features from sorted neighbour distances/labels (shape (n, max_k))."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    max_k = labels.shape[1]

    for k in ks:
        lk = labels[:, :k]
        for c in range(n_classes):
            frac = ((lk == c).sum(axis=1) + smoothing * priors[c]) / (k + smoothing)
            cols.append(frac)
            names.append(f"nn{k}_frac_c{c}")

    # distance-weighted class mass over all max_k neighbours
    w = 1.0 / (dist + EPS)
    wsum = w.sum(axis=1)
    for c in range(n_classes):
        cols.append((w * (labels == c)).sum(axis=1) / wsum)
        names.append(f"nnw_frac_c{c}")

    # distance to the nearest neighbour of each class (fallback: farthest seen)
    for c in range(n_classes):
        dc = np.where(labels == c, dist, np.inf)
        nd = dc.min(axis=1)
        cols.append(np.where(np.isinf(nd), dist[:, -1], nd))
        names.append(f"nn_dist_c{c}")

    # local density proxy: distance to the (min(50, max_k))-th neighbour
    cols.append(dist[:, min(50, max_k) - 1])
    names.append("nn_density")

    return np.stack(cols, axis=1).astype(np.float32), names


def neighbour_features(
    query_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    ref_y: np.ndarray,
    ks: list[int],
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
    max_k: int,
) -> tuple[np.ndarray, list[str]]:
    """Spatial features for ``query_xyz`` using neighbours drawn from ``ref_xyz``."""
    nn = NearestNeighbors(n_neighbors=max_k, n_jobs=-1).fit(ref_xyz)
    dist, idx = nn.kneighbors(query_xyz)
    return _features_from_neighbours(dist, ref_y[idx], ks, n_classes, priors, smoothing)


def oof_neighbour_features(
    xyz: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    ks: list[int],
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
    max_k: int,
) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold spatial features: a row's features use only OTHER folds.

    ``fold_ids`` is passed in explicitly (not derived from ``y`` here) so the
    leakage guarantee is independent of any stratification on the label.
    """
    out: np.ndarray | None = None
    names: list[str] = []
    for fold in np.unique(fold_ids):
        va = fold_ids == fold
        tr = ~va
        feats, names = neighbour_features(
            xyz[va], xyz[tr], y[tr], ks, n_classes, priors, smoothing, max_k
        )
        if out is None:
            out = np.zeros((len(xyz), feats.shape[1]), dtype=np.float32)
        out[va] = feats
    assert out is not None
    return out, names
