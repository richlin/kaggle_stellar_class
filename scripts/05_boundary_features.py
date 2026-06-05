"""Evaluate low-redshift boundary features against the public-best baseline."""
# ruff: noqa: E402
from __future__ import annotations

import itertools
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    append_jsonl,
    balanced_accuracy,
    per_class_recall,
    predict_with_multipliers,
    write_json,
)

REFERENCE_SCORE = 0.9659249816190973
N_SPLITS = 5
SEED = 42
MATERIAL_FOLD_REGRESSION = 0.002
MATERIAL_CLASS_RECALL_REGRESSION = 0.003
STABLE_THRESHOLD_GRID = np.array([0.75, 0.8, 0.9, 1.0, 1.1, 1.15])

RUNS_PATH = Path("experiments/runs.jsonl")
EXPERIMENT_PATH = Path("experiments/05_boundary_v1.json")
OOF_PROB_PATH = Path("experiments/05_boundary_v1_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/05_boundary_v1_test_probabilities.npy")
SUBMISSION_PATH = Path("submissions/05_boundary_v1.csv")

BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 700,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "n_jobs": -1,
    "verbosity": -1,
}


def select_boundary_candidate(candidate_score: float, reference_score: float) -> dict[str, Any]:
    """Accept a boundary candidate only when it improves over the reference OOF."""
    return {
        "accepted": candidate_score > reference_score,
        "candidate_score": float(candidate_score),
        "reference_score": float(reference_score),
        "delta": float(candidate_score - reference_score),
    }


def make_boundary_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def search_stable_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    """Find the best guarded stable-grid multiplier vector."""
    baseline_pred = probabilities.argmax(axis=1)
    baseline_score = balanced_accuracy(y_true, baseline_pred)
    baseline_recall = per_class_recall(y_true, baseline_pred, class_labels)
    baseline_fold_scores = {
        int(fold): balanced_accuracy(y_true[fold_ids == fold], baseline_pred[fold_ids == fold])
        for fold in sorted(np.unique(fold_ids))
    }

    best: dict[str, Any] = {
        "accepted": False,
        "multipliers": np.ones(probabilities.shape[1], dtype=float),
        "baseline_score": baseline_score,
        "score": baseline_score,
        "class_recall_deltas": dict.fromkeys(class_labels, 0.0),
        "fold_score_deltas": {int(fold): 0.0 for fold in sorted(np.unique(fold_ids))},
    }

    for values in itertools.product(STABLE_THRESHOLD_GRID, repeat=probabilities.shape[1]):
        multipliers = np.array(values, dtype=float)
        pred = predict_with_multipliers(probabilities, multipliers)
        score = balanced_accuracy(y_true, pred)
        if score <= best["score"]:
            continue

        recall = per_class_recall(y_true, pred, class_labels)
        class_deltas = {label: recall[label] - baseline_recall[label] for label in class_labels}
        if min(class_deltas.values()) < -MATERIAL_CLASS_RECALL_REGRESSION:
            continue

        fold_deltas = {
            int(fold): balanced_accuracy(y_true[fold_ids == fold], pred[fold_ids == fold])
            - baseline_fold_scores[int(fold)]
            for fold in sorted(np.unique(fold_ids))
        }
        if min(fold_deltas.values()) < -MATERIAL_FOLD_REGRESSION:
            continue

        best = {
            "accepted": True,
            "multipliers": multipliers,
            "baseline_score": baseline_score,
            "score": score,
            "class_recall_deltas": class_deltas,
            "fold_score_deltas": fold_deltas,
        }

    return best


def run_boundary_cv() -> dict[str, Any]:
    """Train a single-seed 5-fold LightGBM with boundary_v1 features."""
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df, feature_set="boundary_v1")
    X_test, _y_test, _test_categorical_columns, _ = build_features(
        test_df,
        feature_set="boundary_v1",
        label_encoder=encoder,
    )
    if y is None:
        raise ValueError("training labels are required")

    class_labels = encoder.classes_.tolist()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), len(class_labels)), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test), len(class_labels)), dtype=np.float64)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    folds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"boundary_v1 fold {fold}/{N_SPLITS}")
        model = LGBMClassifier(**BASE_PARAMS, random_state=SEED + fold)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_columns,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
        )
        valid_probabilities = model.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test) / N_SPLITS
        fold_ids[valid_idx] = fold
        valid_pred = valid_probabilities.argmax(axis=1)
        folds.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_ or BASE_PARAMS["n_estimators"]),
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_pred),
                "per_class_recall": per_class_recall(y[valid_idx], valid_pred, class_labels),
            }
        )

    argmax_pred = oof.argmax(axis=1)
    argmax_score = balanced_accuracy(y, argmax_pred)
    threshold = search_stable_multipliers(y, oof, fold_ids, class_labels)
    chosen_pred = predict_with_multipliers(oof, threshold["multipliers"])
    chosen_score = balanced_accuracy(y, chosen_pred)
    selection = select_boundary_candidate(chosen_score, REFERENCE_SCORE)

    np.save(OOF_PROB_PATH, oof)
    np.save(TEST_PROB_PATH, test_probabilities)
    submission = make_boundary_submission(
        sample_submission,
        test_probabilities,
        threshold["multipliers"],
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "feature_set": "boundary_v1",
        "reference_score": REFERENCE_SCORE,
        "argmax_oof_balanced_accuracy": argmax_score,
        "chosen_oof_balanced_accuracy": chosen_score,
        "candidate_selection": selection,
        "chosen_multipliers": threshold["multipliers"].tolist(),
        "per_class_recall": per_class_recall(y, chosen_pred, class_labels),
        "stable_threshold": {
            **threshold,
            "multipliers": threshold["multipliers"].tolist(),
        },
        "folds": folds,
        "params": BASE_PARAMS,
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "boundary_v1", **record})
    return record


def main() -> int:
    record = run_boundary_cv()
    print(f"boundary_v1 argmax OOF: {record['argmax_oof_balanced_accuracy']:.6f}")
    print(f"boundary_v1 tuned OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs reference: {record['candidate_selection']['delta']:.6f}")
    print(f"accepted: {record['candidate_selection']['accepted']}")
    print(f"multipliers: {record['chosen_multipliers']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
