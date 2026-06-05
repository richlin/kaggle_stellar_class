"""Task 24: graph/cluster spatial residual candidate.

This candidate keeps `16_spatial_blend.csv` as the incumbent and trains a small
residual LightGBM on cached spatial probabilities plus new graph and cluster
class-rate features. Target-dependent features use the same OOF fold ids as the
residual CV so validation rows never see their own fold's labels.
"""
# ruff: noqa: E402
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from src.data import build_features, load_raw
from src.spatial import radec_to_xyz
from src.transductive_spatial import (
    build_probability_meta_features,
    full_train_cluster_class_rates,
    oof_cluster_class_rates,
)
from src.validate import validate_submission
from src.validation import (
    balanced_accuracy,
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
N_SPLITS = 5
FOLD_SEED = 2037
GRAPH_KS = [10, 25, 50, 100, 250]
CLUSTER_SIZES = [512, 2048, 8192]
CLUSTER_RANDOM_STATE = 2041
SPATIAL_BLEND_WEIGHT_LGBM = 0.55

LGBM_PARAMS = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 1100,
    "learning_rate": 0.03,
    "num_leaves": 47,
    "min_child_samples": 40,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "n_jobs": -1,
    "verbosity": -1,
}

SP_LGBM_OOF = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
SP_LGBM_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
SP_XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
SP_XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
SPATIAL_FEATURES_TRAIN = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
SPATIAL_FEATURES_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
SPATIAL_FEATURE_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"

GRAPH_TRAIN = PROJECT_ROOT / "experiments" / "17_graph_train_features.npy"
GRAPH_TEST = PROJECT_ROOT / "experiments" / "17_graph_test_features.npy"
GRAPH_NAMES = PROJECT_ROOT / "experiments" / "17_graph_feature_names.npy"
CLUSTER_TRAIN = PROJECT_ROOT / "experiments" / "17_cluster_train_features.npy"
CLUSTER_TEST = PROJECT_ROOT / "experiments" / "17_cluster_test_features.npy"
CLUSTER_NAMES = PROJECT_ROOT / "experiments" / "17_cluster_feature_names.npy"
OOF_PROB = PROJECT_ROOT / "experiments" / "17_transductive_spatial_oof_probabilities.npy"
TEST_PROB = PROJECT_ROOT / "experiments" / "17_transductive_spatial_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "17_transductive_spatial.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "17_transductive_spatial.csv"


def make_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder,
) -> pd.DataFrame:
    """Build a competition submission while preserving sample id order."""
    predicted = (probabilities * multipliers).argmax(axis=1)
    return pd.DataFrame(
        {
            "id": sample_submission["id"].to_numpy(),
            "class": encoder.inverse_transform(predicted),
        }
    )


def make_fold_ids(y: np.ndarray) -> np.ndarray:
    """Create the single fold assignment shared by target features and CV."""
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    for fold, (_tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_ids[va] = fold
    return fold_ids


def _probabilities_from_neighbours(
    distances: np.ndarray,
    indices: np.ndarray,
    ref_probabilities: np.ndarray,
    k: int,
) -> np.ndarray:
    weights = 1.0 / (distances[:, :k] + 1e-12)
    weighted = (ref_probabilities[indices[:, :k]] * weights[:, :, None]).sum(axis=1)
    return (weighted / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def build_graph_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build OOF train and full-train test graph probability features."""
    if GRAPH_TRAIN.exists() and GRAPH_TEST.exists() and GRAPH_NAMES.exists():
        names = list(np.load(GRAPH_NAMES, allow_pickle=True))
        return np.load(GRAPH_TRAIN), np.load(GRAPH_TEST), names

    train_xyz = radec_to_xyz(train["alpha"].to_numpy(), train["delta"].to_numpy())
    test_xyz = radec_to_xyz(test["alpha"].to_numpy(), test["delta"].to_numpy())
    one_hot = np.eye(len(CLASS_LABELS), dtype=np.float32)[y]
    max_k = max(GRAPH_KS)
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []

    train_out_by_k = {
        k: np.zeros((len(train), len(CLASS_LABELS)), dtype=np.float32) for k in GRAPH_KS
    }
    for fold in np.unique(fold_ids):
        valid = fold_ids == fold
        fit = ~valid
        print(f"  graph fold {fold + 1}/{N_SPLITS}")
        nn = NearestNeighbors(n_neighbors=max_k, n_jobs=-1).fit(train_xyz[fit])
        distances, indices = nn.kneighbors(train_xyz[valid])
        ref_probabilities = one_hot[fit]
        for k in GRAPH_KS:
            train_out_by_k[k][valid] = _probabilities_from_neighbours(
                distances,
                indices,
                ref_probabilities,
                k,
            )

    nn = NearestNeighbors(n_neighbors=max_k, n_jobs=-1).fit(train_xyz)
    test_distances, test_indices = nn.kneighbors(test_xyz)
    for k in GRAPH_KS:
        block = train_out_by_k[k]
        train_blocks.append(block)
        test_blocks.append(_probabilities_from_neighbours(test_distances, test_indices, one_hot, k))
        names.extend([f"graph{k}_p{class_idx}" for class_idx in range(len(CLASS_LABELS))])

    train_features = np.concatenate(train_blocks, axis=1)
    test_features = np.concatenate(test_blocks, axis=1)
    np.save(GRAPH_TRAIN, train_features)
    np.save(GRAPH_TEST, test_features)
    np.save(GRAPH_NAMES, np.array(names, dtype=object))
    return train_features, test_features, names


def build_cluster_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build multi-resolution cluster target-rate features."""
    if CLUSTER_TRAIN.exists() and CLUSTER_TEST.exists() and CLUSTER_NAMES.exists():
        names = list(np.load(CLUSTER_NAMES, allow_pickle=True))
        return np.load(CLUSTER_TRAIN), np.load(CLUSTER_TEST), names

    train_xyz = radec_to_xyz(train["alpha"].to_numpy(), train["delta"].to_numpy())
    test_xyz = radec_to_xyz(test["alpha"].to_numpy(), test["delta"].to_numpy())
    all_xyz = np.vstack([train_xyz, test_xyz])
    priors = np.bincount(y, minlength=len(CLASS_LABELS)) / len(y)
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    names: list[str] = []

    for n_clusters in CLUSTER_SIZES:
        print(f"  clustering {n_clusters} spatial cells")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=CLUSTER_RANDOM_STATE + n_clusters,
            batch_size=65536,
            n_init=1,
            max_iter=60,
            reassignment_ratio=0.0,
        )
        cluster_ids = kmeans.fit_predict(all_xyz)
        train_ids = cluster_ids[: len(train)]
        test_ids = cluster_ids[len(train):]
        smoothing = max(10.0, len(train) / n_clusters * 0.25)
        train_rates = oof_cluster_class_rates(
            train_ids,
            y,
            fold_ids,
            n_clusters=n_clusters,
            n_classes=len(CLASS_LABELS),
            priors=priors,
            smoothing=smoothing,
        )
        test_rates = full_train_cluster_class_rates(
            train_ids,
            test_ids,
            y,
            n_clusters=n_clusters,
            n_classes=len(CLASS_LABELS),
            priors=priors,
            smoothing=smoothing,
        )
        train_blocks.append(train_rates)
        test_blocks.append(test_rates)
        names.extend([f"cluster{n_clusters}_p{class_idx}" for class_idx in range(len(CLASS_LABELS))])

    train_features = np.concatenate(train_blocks, axis=1)
    test_features = np.concatenate(test_blocks, axis=1)
    np.save(CLUSTER_TRAIN, train_features)
    np.save(CLUSTER_TEST, test_features)
    np.save(CLUSTER_NAMES, np.array(names, dtype=object))
    return train_features, test_features, names


def add_array_features(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    train_values: np.ndarray,
    test_values: np.ndarray,
    names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = pd.DataFrame(train_values, columns=names, index=X.index)
    test_frame = pd.DataFrame(test_values, columns=names, index=X_test.index)
    return pd.concat([X, train_frame], axis=1), pd.concat([X_test, test_frame], axis=1)


def load_spatial_probability_blocks() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    lgbm_oof = np.load(SP_LGBM_OOF)
    lgbm_test = np.load(SP_LGBM_TEST)
    xgb_oof = np.load(SP_XGB_OOF)
    xgb_test = np.load(SP_XGB_TEST)
    blend_oof = SPATIAL_BLEND_WEIGHT_LGBM * lgbm_oof + (1 - SPATIAL_BLEND_WEIGHT_LGBM) * xgb_oof
    blend_test = SPATIAL_BLEND_WEIGHT_LGBM * lgbm_test + (1 - SPATIAL_BLEND_WEIGHT_LGBM) * xgb_test
    return (
        {"spatial_lgbm": lgbm_oof, "spatial_xgb": xgb_oof, "spatial_blend": blend_oof},
        {"spatial_lgbm": lgbm_test, "spatial_xgb": xgb_test, "spatial_blend": blend_test},
    )


def build_model_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    X, _y, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    spatial_names = list(np.load(SPATIAL_FEATURE_NAMES, allow_pickle=True))
    X, X_test = add_array_features(
        X,
        X_test,
        np.load(SPATIAL_FEATURES_TRAIN),
        np.load(SPATIAL_FEATURES_TEST),
        [f"spatial_{name}" for name in spatial_names],
    )

    graph_train, graph_test, graph_names = build_graph_features(train, test, y, fold_ids)
    X, X_test = add_array_features(X, X_test, graph_train, graph_test, graph_names)

    cluster_train, cluster_test, cluster_names = build_cluster_features(train, test, y, fold_ids)
    X, X_test = add_array_features(X, X_test, cluster_train, cluster_test, cluster_names)

    train_blocks, test_blocks = load_spatial_probability_blocks()
    train_meta, test_meta = build_probability_meta_features(train_blocks, test_blocks)
    X = pd.concat([X, train_meta], axis=1)
    X_test = pd.concat([X_test, test_meta], axis=1)
    return X, X_test, cat_cols, train_blocks["spatial_blend"], test_blocks["spatial_blend"]


def run_residual_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(X), len(CLASS_LABELS)), dtype=float)
    test_prob = np.zeros((len(X_test), len(CLASS_LABELS)), dtype=float)
    for fold in np.unique(fold_ids):
        valid = fold_ids == fold
        train = ~valid
        print(f"  residual fold {fold + 1}/{N_SPLITS}")
        model = LGBMClassifier(**LGBM_PARAMS, random_state=FOLD_SEED + int(fold))
        model.fit(
            X.loc[train],
            y[train],
            eval_set=[(X.loc[valid], y[valid])],
            eval_metric="multi_logloss",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
        )
        oof[valid] = model.predict_proba(X.loc[valid])
        test_prob += model.predict_proba(X_test) / N_SPLITS
    return oof, test_prob


def search_blend(
    y: np.ndarray,
    incumbent_oof: np.ndarray,
    residual_oof: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    best_weight = 0.0
    best_multipliers = np.ones(len(CLASS_LABELS))
    best_score = balanced_accuracy(y, incumbent_oof.argmax(axis=1))
    best_probabilities = incumbent_oof
    for residual_weight in np.linspace(0.0, 0.8, 33):
        probabilities = (1 - residual_weight) * incumbent_oof + residual_weight * residual_oof
        multipliers, score = search_class_multipliers(y, probabilities)
        if score > best_score:
            best_weight = float(residual_weight)
            best_multipliers = multipliers
            best_score = score
            best_probabilities = probabilities
    return best_weight, best_multipliers, best_score, best_probabilities


def main() -> int:
    train, test, sample = load_raw()
    X_base, y, _cat, encoder = build_features(train)
    if y is None:
        raise ValueError("training data must include class labels")
    del X_base

    fold_ids = make_fold_ids(y)
    X, X_test, cat_cols, incumbent_oof, incumbent_test = build_model_frames(train, test, y, fold_ids)
    print(f"residual feature matrix: {X.shape}")
    residual_oof, residual_test = run_residual_cv(X, y, X_test, cat_cols, fold_ids)
    np.save(OOF_PROB, residual_oof)
    np.save(TEST_PROB, residual_test)

    residual_multipliers, residual_score = search_class_multipliers(y, residual_oof)
    incumbent_multipliers = np.array([0.6, 1.0, 1.35])
    incumbent_score = balanced_accuracy(y, (incumbent_oof * incumbent_multipliers).argmax(axis=1))
    blend_weight, blend_multipliers, blend_score, blend_oof = search_blend(
        y,
        incumbent_oof,
        residual_oof,
    )
    blend_test = (1 - blend_weight) * incumbent_test + blend_weight * residual_test
    blend_pred = (blend_oof * blend_multipliers).argmax(axis=1)
    recalls = per_class_recall(y, blend_pred, CLASS_LABELS)

    submission = make_submission(sample, blend_test, blend_multipliers, encoder)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)

    write_json(
        EXPERIMENT,
        {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
            "fold_seed": FOLD_SEED,
            "graph_ks": GRAPH_KS,
            "cluster_sizes": CLUSTER_SIZES,
            "params": LGBM_PARAMS,
            "incumbent_oof": incumbent_score,
            "residual_tuned_oof": residual_score,
            "residual_multipliers": residual_multipliers.tolist(),
            "blend_weight_residual": blend_weight,
            "blend_tuned_oof": blend_score,
            "blend_multipliers": blend_multipliers.tolist(),
            "blend_per_class_recall": recalls,
            "confusion_matrix": confusion_matrix(y, blend_pred).tolist(),
            "submission_path": str(SUBMISSION),
        },
    )

    print("\n================ TRANSDUCTIVE SPATIAL RESULT ================")
    print(f"incumbent OOF       : {incumbent_score:.6f}")
    print(f"residual tuned OOF  : {residual_score:.6f}")
    print(f"best blend OOF      : {blend_score:.6f} (residual weight={blend_weight:.3f})")
    print(f"blend recalls       : {recalls}")
    print(f"blend multipliers   : {blend_multipliers.round(4).tolist()}")
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
