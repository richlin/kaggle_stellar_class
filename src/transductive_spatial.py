"""Task 24 graph and cluster features for spatial residual modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


def weighted_graph_probabilities(
    query_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    ref_probabilities: np.ndarray,
    n_neighbors: int,
    self_reference_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return inverse-distance weighted class probabilities from a spatial graph.

    ``self_reference_indices`` is used when query rows are also present in the
    reference set. Matching reference rows are zero-weighted so a test row cannot
    copy its own soft label back into its graph feature.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if len(ref_xyz) != len(ref_probabilities):
        raise ValueError("ref_xyz and ref_probabilities must have the same row count")

    extra = 1 if self_reference_indices is not None else 0
    neighbours_to_fetch = min(n_neighbors + extra, len(ref_xyz))
    nn = NearestNeighbors(n_neighbors=neighbours_to_fetch, n_jobs=-1).fit(ref_xyz)
    distances, indices = nn.kneighbors(query_xyz)

    weights = 1.0 / (distances + EPS)
    if self_reference_indices is not None:
        if self_reference_indices.shape != (len(query_xyz),):
            raise ValueError("self_reference_indices must have one entry per query row")
        weights = weights.copy()
        weights[indices == self_reference_indices[:, None]] = 0.0

    weight_sums = weights.sum(axis=1, keepdims=True)
    fallback = ref_probabilities.mean(axis=0)
    weighted = (ref_probabilities[indices] * weights[:, :, None]).sum(axis=1)
    out = np.divide(weighted, weight_sums, out=np.zeros_like(weighted), where=weight_sums > 0)
    empty = weight_sums[:, 0] <= 0
    if empty.any():
        out[empty] = fallback
    return out.astype(np.float32)


def _cluster_rates(
    cluster_ids: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    counts = np.zeros((n_clusters, n_classes), dtype=float)
    np.add.at(counts, (cluster_ids, y), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return (counts + smoothing * priors) / (totals + smoothing)


def oof_cluster_class_rates(
    cluster_ids: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    n_clusters: int,
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Build OOF-safe cluster class rates for train rows."""
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    for fold in np.unique(fold_ids):
        valid = fold_ids == fold
        train = ~valid
        rates = _cluster_rates(
            cluster_ids[train],
            y[train],
            n_clusters=n_clusters,
            n_classes=n_classes,
            priors=priors,
            smoothing=smoothing,
        )
        out[valid] = rates[cluster_ids[valid]]
    return out


def full_train_cluster_class_rates(
    train_cluster_ids: np.ndarray,
    test_cluster_ids: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    n_classes: int,
    priors: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Build full-train cluster class rates for test rows."""
    rates = _cluster_rates(
        train_cluster_ids,
        y,
        n_clusters=n_clusters,
        n_classes=n_classes,
        priors=priors,
        smoothing=smoothing,
    )
    return rates[test_cluster_ids].astype(np.float32)


def _probability_meta_frame(probabilities: np.ndarray, prefix: str) -> pd.DataFrame:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    clipped = np.clip(probabilities, EPS, 1.0)
    order = np.sort(probabilities, axis=1)
    data: dict[str, np.ndarray] = {}
    for class_idx in range(probabilities.shape[1]):
        data[f"{prefix}_p{class_idx}"] = probabilities[:, class_idx]
    data[f"{prefix}_margin"] = order[:, -1] - order[:, -2]
    data[f"{prefix}_max_prob"] = order[:, -1]
    data[f"{prefix}_entropy"] = -(clipped * np.log(clipped)).sum(axis=1)
    for left in range(probabilities.shape[1]):
        for right in range(left + 1, probabilities.shape[1]):
            data[f"{prefix}_gap_{left}_{right}"] = probabilities[:, left] - probabilities[:, right]
    data[f"{prefix}_pred"] = probabilities.argmax(axis=1).astype(float)
    return pd.DataFrame(data, dtype=np.float32)


def build_probability_meta_features(
    train_probability_blocks: dict[str, np.ndarray],
    test_probability_blocks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create aligned meta-features from named train/test probability blocks."""
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    if train_probability_blocks.keys() != test_probability_blocks.keys():
        raise ValueError("train and test probability blocks must have the same names")

    for name in train_probability_blocks:
        train_prob = train_probability_blocks[name]
        test_prob = test_probability_blocks[name]
        if train_prob.shape[1] != test_prob.shape[1]:
            raise ValueError(f"probability block {name!r} has mismatched class counts")
        train_frames.append(_probability_meta_frame(train_prob, name))
        test_frames.append(_probability_meta_frame(test_prob, name))

    train_meta = pd.concat(train_frames, axis=1)
    test_meta = pd.concat(test_frames, axis=1)
    return train_meta, test_meta
