"""Phase 5 model-diversity ensemble and submission generation.

The cheap helper functions are tested directly. The training commands are
resumable: each candidate writes OOF/test probabilities that the blend step can
reuse without retraining.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    append_jsonl,
    balanced_accuracy,
    predict_with_multipliers,
    search_class_multipliers,
    write_json,
)

N_SPLITS = 5
SEED = 42
MATERIAL_FOLD_REGRESSION = 0.002
MATERIAL_CLASS_RECALL_REGRESSION = 0.003
STABLE_THRESHOLD_GRID = np.array([0.75, 0.8, 0.9, 1.0, 1.1, 1.15])

RUNS_PATH = Path("experiments/runs.jsonl")
EXPERIMENT_PATH = Path("experiments/04_ensemble.json")
SUBMISSION_PATH = Path("submissions/04_ensemble.csv")
REFERENCE_OOF_PATH = Path("experiments/03_final_oof_probabilities.npy")
REFERENCE_TEST_PATH = Path("experiments/03_final_test_probabilities.npy")

LGBM_DART_PARAMS: dict[str, Any] = {
    "boosting_type": "dart",
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 900,
    "learning_rate": 0.035,
    "num_leaves": 47,
    "min_child_samples": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 0.2,
    "n_jobs": -1,
    "verbosity": -1,
}

XGB_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 1200,
    "learning_rate": 0.04,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "tree_method": "hist",
    "n_jobs": -1,
}

CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "MultiClass",
    "iterations": 1200,
    "learning_rate": 0.04,
    "depth": 8,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "allow_writing_files": False,
}


def weighted_probability_blend(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return a weighted average of compatible probability matrices."""
    if not probabilities:
        raise ValueError("at least one probability matrix is required")

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(probabilities),):
        raise ValueError(f"weights shape must be {(len(probabilities),)}, got {weights.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("weights must be finite and sum to a positive value")

    first_shape = probabilities[0].shape
    for probability in probabilities:
        if probability.shape != first_shape:
            raise ValueError("all probability matrices must have the same shape")

    normalized = weights / weights.sum()
    blended = np.zeros_like(probabilities[0], dtype=float)
    for probability, weight in zip(probabilities, normalized, strict=True):
        blended += probability * weight
    return blended


def make_ensemble_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def _fast_recall_array(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    confusion = np.bincount(
        (y_true * n_classes) + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    support = confusion.sum(axis=1)
    return np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(n_classes, dtype=float),
        where=support != 0,
    )


def _fast_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(_fast_recall_array(y_true, y_pred, n_classes).mean())


def _fast_per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> dict[str, float]:
    recalls = _fast_recall_array(y_true, y_pred, len(class_labels))
    return {label: float(score) for label, score in zip(class_labels, recalls, strict=True)}


def _fold_scores(y_true: np.ndarray, y_pred: np.ndarray, fold_ids: np.ndarray) -> dict[int, float]:
    n_classes = int(np.max(y_true)) + 1
    return {
        int(fold): _fast_balanced_accuracy(
            y_true[fold_ids == fold],
            y_pred[fold_ids == fold],
            n_classes,
        )
        for fold in sorted(np.unique(fold_ids))
    }


def _stability_deltas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_recall: dict[str, float],
    baseline_fold_scores: dict[int, float],
    fold_ids: np.ndarray,
    class_labels: Sequence[str],
) -> tuple[dict[str, float], dict[int, float]]:
    recall = _fast_per_class_recall(y_true, y_pred, class_labels)
    fold_scores = _fold_scores(y_true, y_pred, fold_ids)
    class_deltas = {label: recall[label] - baseline_recall[label] for label in class_labels}
    fold_deltas = {fold: fold_scores[fold] - baseline_fold_scores[fold] for fold in fold_scores}
    return class_deltas, fold_deltas


def _is_stable(class_deltas: dict[str, float], fold_deltas: dict[int, float]) -> bool:
    return (
        min(class_deltas.values()) >= -MATERIAL_CLASS_RECALL_REGRESSION
        and min(fold_deltas.values()) >= -MATERIAL_FOLD_REGRESSION
    )


def _search_stable_grid_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
    evaluate: Any,
    extra_candidate: np.ndarray | None,
) -> dict[str, Any]:
    """Exhaustively search the same guarded coarse grid used by Phase 4."""
    baseline = evaluate(np.ones(probabilities.shape[1], dtype=float))
    best = baseline
    candidates = [
        np.array(values, dtype=float)
        for values in itertools.product(STABLE_THRESHOLD_GRID, repeat=probabilities.shape[1])
    ]
    if extra_candidate is not None:
        candidates.append(np.asarray(extra_candidate, dtype=float))

    for multipliers in candidates:
        result = evaluate(multipliers)
        if result["stable"] and result["score"] > best["score"]:
            best = result
    return best


def search_continuous_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
    initial_multipliers: np.ndarray | None = None,
) -> dict[str, Any]:
    """Optimize class multipliers continuously, guarded by fold/class stability."""
    baseline_pred = probabilities.argmax(axis=1)
    baseline_score = _fast_balanced_accuracy(y_true, baseline_pred, len(class_labels))
    baseline_recall = _fast_per_class_recall(y_true, baseline_pred, class_labels)
    baseline_fold_scores = _fold_scores(y_true, baseline_pred, fold_ids)

    def evaluate(multipliers: np.ndarray) -> dict[str, Any]:
        pred = predict_with_multipliers(probabilities, multipliers)
        score = _fast_balanced_accuracy(y_true, pred, len(class_labels))
        class_deltas, fold_deltas = _stability_deltas(
            y_true,
            pred,
            baseline_recall,
            baseline_fold_scores,
            fold_ids,
            class_labels,
        )
        return {
            "multipliers": multipliers,
            "score": score,
            "per_class_recall": _fast_per_class_recall(y_true, pred, class_labels),
            "class_recall_deltas": class_deltas,
            "fold_score_deltas": fold_deltas,
            "stable": _is_stable(class_deltas, fold_deltas),
        }

    if initial_multipliers is None:
        initial_multipliers, _score = search_class_multipliers(y_true, probabilities)
    coarse = _search_stable_grid_multipliers(
        y_true,
        probabilities,
        fold_ids,
        class_labels,
        evaluate,
        initial_multipliers,
    )

    def objective(log_multipliers: np.ndarray) -> float:
        multipliers = np.exp(log_multipliers)
        result = evaluate(multipliers)
        penalty = 0.0 if result["stable"] else 1.0
        return -result["score"] + penalty

    optimizer = minimize(
        objective,
        np.log(np.clip(coarse["multipliers"], 1e-6, None)),
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-7},
    )
    continuous = evaluate(np.exp(optimizer.x))

    selected = continuous if continuous["stable"] and continuous["score"] >= coarse["score"] else coarse
    return {
        "accepted": bool(selected["stable"]),
        "method": "continuous" if selected is continuous else "coarse_or_baseline",
        "baseline_score": baseline_score,
        "score": selected["score"],
        "multipliers": selected["multipliers"],
        "per_class_recall": selected["per_class_recall"],
        "class_recall_deltas": selected["class_recall_deltas"],
        "fold_score_deltas": selected["fold_score_deltas"],
        "optimizer_success": bool(optimizer.success),
    }


def _weight_grid(n_models: int, step: float = 0.1) -> list[np.ndarray]:
    values = np.round(np.arange(0.0, 1.0 + step, step), 10)
    if n_models == 1:
        return [np.ones(1)]
    if n_models == 2:
        return [np.array([left, 1.0 - left], dtype=float) for left in values]

    weights = []
    for candidate in itertools.product(values, repeat=n_models):
        if abs(sum(candidate) - 1.0) <= 1e-9 and any(value > 0 for value in candidate):
            weights.append(np.array(candidate, dtype=float))
    return weights


def search_blend_weights(
    y_true: np.ndarray,
    probability_sets: list[np.ndarray],
    fold_ids: np.ndarray,
    class_labels: list[str],
    max_threshold_candidates: int = 12,
) -> dict[str, Any]:
    """Grid-search blend weights, then tune class multipliers for each blend."""
    best: dict[str, Any] | None = None
    scored_weights: list[tuple[float, np.ndarray]] = []

    for weights in _weight_grid(len(probability_sets)):
        blended = weighted_probability_blend(probability_sets, weights)
        argmax_score = _fast_balanced_accuracy(y_true, blended.argmax(axis=1), len(class_labels))
        scored_weights.append((argmax_score, weights))

    scored_weights.sort(key=lambda item: item[0], reverse=True)
    threshold_weights = scored_weights
    if len(scored_weights) > max_threshold_candidates:
        endpoints = [
            (score, weights)
            for score, weights in scored_weights
            if np.count_nonzero(weights) == 1
        ]
        threshold_weights = scored_weights[:max_threshold_candidates] + endpoints

    seen_weights: set[tuple[float, ...]] = set()
    for argmax_score, weights in threshold_weights:
        key = tuple(weights.round(10).tolist())
        if key in seen_weights:
            continue
        seen_weights.add(key)
        blended = weighted_probability_blend(probability_sets, weights)
        threshold = search_continuous_multipliers(y_true, blended, fold_ids, class_labels)
        result = {
            "weights": weights,
            "multipliers": threshold["multipliers"],
            "score": threshold["score"],
            "argmax_score": argmax_score,
            "per_class_recall": threshold["per_class_recall"],
            "threshold": threshold,
        }
        if best is None or result["score"] > best["score"]:
            best = result

    if best is None:
        raise ValueError("no blend candidates were evaluated")
    return best


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _candidate_paths(candidate: str) -> tuple[Path, Path]:
    return (
        Path(f"experiments/04_{candidate}_oof_probabilities.npy"),
        Path(f"experiments/04_{candidate}_test_probabilities.npy"),
    )


def _load_training_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, list[str], LabelEncoder]:
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    if y is None:
        raise ValueError("training labels are required")
    return X, y, X_test, sample_submission, categorical_columns, encoder


def _make_fold_ids(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (_train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        fold_ids[valid_idx] = fold
    return fold_ids


def _xgboost_frames(X: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([X, X_test], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined, columns=combined.select_dtypes(["category"]).columns)
    return encoded.iloc[: len(X)].astype(float), encoded.iloc[len(X) :].astype(float)


def _train_lgbm_dart(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    class_labels: list[str],
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(class_labels)), dtype=np.float64)
    test = np.zeros((len(X_test), len(class_labels)), dtype=np.float64)
    folds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"lgbm_dart fold {fold}/{N_SPLITS}")
        model = LGBMClassifier(**LGBM_DART_PARAMS, random_state=SEED + fold)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_columns,
            callbacks=[log_evaluation(period=0)],
        )
        valid_prob = model.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = valid_prob
        test += model.predict_proba(X_test) / N_SPLITS
        folds.append(
            {
                "fold": fold,
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_prob.argmax(axis=1)),
            }
        )

    return {"oof": oof, "test": test, "folds": folds, "params": LGBM_DART_PARAMS}


def _train_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    class_labels: list[str],
) -> dict[str, Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("xgboost is not installed") from exc

    X_encoded, X_test_encoded = _xgboost_frames(X, X_test)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X_encoded), len(class_labels)), dtype=np.float64)
    test = np.zeros((len(X_test_encoded), len(class_labels)), dtype=np.float64)
    folds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_encoded, y), start=1):
        print(f"xgboost fold {fold}/{N_SPLITS}")
        model = XGBClassifier(**XGB_PARAMS, random_state=SEED + fold)
        model.fit(
            X_encoded.iloc[train_idx],
            y[train_idx],
            sample_weight=compute_sample_weight("balanced", y[train_idx]),
            eval_set=[(X_encoded.iloc[valid_idx], y[valid_idx])],
            verbose=False,
        )
        valid_prob = model.predict_proba(X_encoded.iloc[valid_idx])
        oof[valid_idx] = valid_prob
        test += model.predict_proba(X_test_encoded) / N_SPLITS
        folds.append(
            {
                "fold": fold,
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_prob.argmax(axis=1)),
            }
        )

    return {"oof": oof, "test": test, "folds": folds, "params": XGB_PARAMS}


def _train_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    class_labels: list[str],
) -> dict[str, Any]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("catboost is not installed") from exc

    cat_features = [X.columns.get_loc(column) for column in categorical_columns]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(class_labels)), dtype=np.float64)
    test = np.zeros((len(X_test), len(class_labels)), dtype=np.float64)
    folds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"catboost fold {fold}/{N_SPLITS}")
        model = CatBoostClassifier(**{**CATBOOST_PARAMS, "random_seed": SEED + fold})
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            cat_features=cat_features,
            eval_set=(X.iloc[valid_idx], y[valid_idx]),
            use_best_model=True,
        )
        valid_prob = model.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = valid_prob
        test += model.predict_proba(X_test) / N_SPLITS
        folds.append(
            {
                "fold": fold,
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_prob.argmax(axis=1)),
            }
        )

    return {"oof": oof, "test": test, "folds": folds, "params": CATBOOST_PARAMS}


def train_candidate(candidate: str) -> dict[str, Any]:
    """Train one Phase 5 candidate and save compatible probability arrays."""
    X, y, X_test, _sample_submission, categorical_columns, encoder = _load_training_data()
    class_labels = encoder.classes_.tolist()

    if candidate == "lgbm_dart":
        result = _train_lgbm_dart(X, y, X_test, categorical_columns, class_labels)
    elif candidate == "xgboost":
        result = _train_xgboost(X, y, X_test, class_labels)
    elif candidate == "catboost":
        result = _train_catboost(X, y, X_test, categorical_columns, class_labels)
    else:
        raise ValueError("candidate must be one of: lgbm_dart, xgboost, catboost")

    oof_path, test_path = _candidate_paths(candidate)
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    fold_ids = _make_fold_ids(X, y)
    threshold = search_continuous_multipliers(y, result["oof"], fold_ids, class_labels)
    record = {
        "kind": "phase5_candidate",
        "timestamp_utc": _timestamp(),
        "candidate": candidate,
        "argmax_oof_balanced_accuracy": balanced_accuracy(y, result["oof"].argmax(axis=1)),
        "chosen_oof_balanced_accuracy": threshold["score"],
        "chosen_multipliers": threshold["multipliers"].tolist(),
        "per_class_recall": threshold["per_class_recall"],
        "folds": result["folds"],
        "params": result["params"],
        "oof_probability_path": str(oof_path),
        "test_probability_path": str(test_path),
    }
    append_jsonl(RUNS_PATH, record)
    print(f"{candidate} argmax OOF: {record['argmax_oof_balanced_accuracy']:.6f}")
    print(f"{candidate} tuned OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"wrote {oof_path}")
    print(f"wrote {test_path}")
    return record


def _load_available_probability_sets() -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    names = []
    oof_sets = []
    test_sets = []
    candidates = [
        ("lgbm_seed_average_final", REFERENCE_OOF_PATH, REFERENCE_TEST_PATH),
        ("xgboost", *_candidate_paths("xgboost")),
        ("catboost", *_candidate_paths("catboost")),
        ("lgbm_dart", *_candidate_paths("lgbm_dart")),
    ]
    for name, oof_path, test_path in candidates:
        if oof_path.exists() and test_path.exists():
            names.append(name)
            oof_sets.append(np.load(oof_path))
            test_sets.append(np.load(test_path))
    if not names:
        raise FileNotFoundError("no compatible OOF/test probability arrays found")
    return names, oof_sets, test_sets


def blend_available_candidates() -> dict[str, Any]:
    """Blend all available probability arrays and write the Phase 5 submission."""
    X, y, _X_test, sample_submission, _categorical_columns, encoder = _load_training_data()
    class_labels = encoder.classes_.tolist()
    fold_ids = _make_fold_ids(X, y)
    names, oof_sets, test_sets = _load_available_probability_sets()

    best = search_blend_weights(y, oof_sets, fold_ids, class_labels)
    blended_test = weighted_probability_blend(test_sets, best["weights"])
    submission = make_ensemble_submission(
        sample_submission,
        blended_test,
        best["multipliers"],
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record = {
        "timestamp_utc": _timestamp(),
        "candidate_names": names,
        "weights": best["weights"].tolist(),
        "argmax_oof_balanced_accuracy": best["argmax_score"],
        "chosen_oof_balanced_accuracy": best["score"],
        "chosen_multipliers": best["multipliers"].tolist(),
        "per_class_recall": best["per_class_recall"],
        "threshold": {
            **best["threshold"],
            "multipliers": best["threshold"]["multipliers"].tolist(),
        },
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    print(f"available candidates: {', '.join(names)}")
    print(f"weights: {record['weights']}")
    print(f"argmax OOF: {record['argmax_oof_balanced_accuracy']:.6f}")
    print(f"tuned OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"multipliers: {record['chosen_multipliers']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-candidate",
        choices=["lgbm_dart", "xgboost", "catboost"],
        help="train and save one base-learner probability set",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="blend available saved probability sets and write the ensemble submission",
    )
    args = parser.parse_args(argv)

    if args.train_candidate:
        train_candidate(args.train_candidate)
        return 0
    if args.blend:
        blend_available_candidates()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
