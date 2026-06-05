"""Spatial features that use audited external labelled rows as neighbours."""
from __future__ import annotations

import numpy as np

from src.spatial import neighbour_features


def make_append_sample_weights(
    n_competition: int,
    n_external: int,
    external_weight: float,
) -> np.ndarray:
    """Return training sample weights for competition + external appended rows."""
    if external_weight <= 0:
        raise ValueError("external_weight must be positive")
    return np.concatenate([
        np.ones(n_competition, dtype=float),
        np.full(n_external, external_weight, dtype=float),
    ])


def external_reference_oof_features(
    comp_xyz: np.ndarray,
    comp_y: np.ndarray,
    fold_ids: np.ndarray,
    external_xyz: np.ndarray,
    external_y: np.ndarray,
    ks: list[int],
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
    max_k: int,
) -> tuple[np.ndarray, list[str]]:
    """OOF spatial features using train-fold competition rows plus external rows.

    For each competition validation fold, labelled neighbours are drawn from all
    other competition folds plus every audited external labelled row. Validation
    rows never use their own labels.
    """
    if len(comp_xyz) != len(comp_y) or len(comp_y) != len(fold_ids):
        raise ValueError("competition xyz, labels, and fold ids must have the same length")
    if len(external_xyz) != len(external_y):
        raise ValueError("external xyz and labels must have the same length")
    if len(external_y) == 0:
        raise ValueError("external rows are required for external-reference features")

    out: np.ndarray | None = None
    names: list[str] = []
    for fold in np.unique(fold_ids):
        va = fold_ids == fold
        tr = ~va
        ref_xyz = np.vstack([comp_xyz[tr], external_xyz])
        ref_y = np.concatenate([comp_y[tr], external_y])
        feats, names = neighbour_features(
            comp_xyz[va],
            ref_xyz,
            ref_y,
            ks,
            n_classes,
            priors,
            smoothing,
            max_k,
        )
        if out is None:
            out = np.zeros((len(comp_xyz), feats.shape[1]), dtype=np.float32)
        out[va] = feats

    assert out is not None
    return out, names


def external_reference_test_features(
    query_xyz: np.ndarray,
    comp_xyz: np.ndarray,
    comp_y: np.ndarray,
    external_xyz: np.ndarray,
    external_y: np.ndarray,
    ks: list[int],
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
    max_k: int,
) -> tuple[np.ndarray, list[str]]:
    """Test-time spatial features using all competition and external labels."""
    ref_xyz = np.vstack([comp_xyz, external_xyz])
    ref_y = np.concatenate([comp_y, external_y])
    return neighbour_features(query_xyz, ref_xyz, ref_y, ks, n_classes, priors, smoothing, max_k)
