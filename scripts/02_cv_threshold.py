"""Run 5-fold LightGBM CV, tune class multipliers, and write a tuned submission."""
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
    balanced_accuracy,
    per_class_recall,
    predict_with_multipliers,
    search_class_multipliers,
    write_json,
)

SEED = 42
N_SPLITS = 5
MATERIAL_FOLD_REGRESSION = 0.002
MATERIAL_CLASS_RECALL_REGRESSION = 0.003
OOF_PROB_PATH = Path("experiments/02_cv_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/02_cv_test_probabilities.npy")
SUBMISSION_PATH = Path("submissions/02_cv_tuned.csv")
EXPERIMENT_PATH = Path("experiments/02_cv_threshold.json")
MODEL_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 700,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def make_tuned_submission(
    ids: pd.Series,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission DataFrame from probabilities and class multipliers."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": ids.to_numpy(), "class": labels})


def recall_variation_warnings(
    fold_recalls: list[dict[str, float]],
    max_allowed_range: float = 0.02,
) -> dict[str, float]:
    """Return classes whose recall range across folds exceeds ``max_allowed_range``."""
    if not fold_recalls:
        return {}

    warnings: dict[str, float] = {}
    for label in fold_recalls[0]:
        values = [recall[label] for recall in fold_recalls]
        value_range = max(values) - min(values)
        if value_range > max_allowed_range:
            warnings[label] = value_range
    return warnings


def search_stable_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
    grid: np.ndarray | None = None,
    min_class_recall_delta: float = -MATERIAL_CLASS_RECALL_REGRESSION,
    min_fold_score_delta: float = -MATERIAL_FOLD_REGRESSION,
) -> dict[str, Any]:
    """Find the best multiplier vector that respects class and fold stability limits."""
    if grid is None:
        grid = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])

    baseline_pred = probabilities.argmax(axis=1)
    baseline_score = balanced_accuracy(y_true, baseline_pred)
    baseline_recall = per_class_recall(y_true, baseline_pred, class_labels)
    baseline_fold_scores = {
        int(fold): balanced_accuracy(y_true[fold_ids == fold], baseline_pred[fold_ids == fold])
        for fold in sorted(np.unique(fold_ids))
    }

    best: dict[str, Any] | None = None
    for values in itertools.product(grid, repeat=probabilities.shape[1]):
        multipliers = np.array(values, dtype=float)
        pred = predict_with_multipliers(probabilities, multipliers)
        score = balanced_accuracy(y_true, pred)
        if score <= baseline_score:
            continue

        recall = per_class_recall(y_true, pred, class_labels)
        class_recall_deltas = {
            label: recall[label] - baseline_recall[label] for label in class_labels
        }
        if min(class_recall_deltas.values()) < min_class_recall_delta:
            continue

        fold_score_deltas = {
            int(fold): balanced_accuracy(y_true[fold_ids == fold], pred[fold_ids == fold])
            - baseline_fold_scores[int(fold)]
            for fold in sorted(np.unique(fold_ids))
        }
        if min(fold_score_deltas.values()) < min_fold_score_delta:
            continue

        candidate = {
            "accepted": True,
            "multipliers": multipliers,
            "baseline_score": baseline_score,
            "score": score,
            "class_recall_deltas": class_recall_deltas,
            "fold_score_deltas": fold_score_deltas,
        }
        if best is None or score > best["score"]:
            best = candidate

    if best is not None:
        return best

    return {
        "accepted": False,
        "multipliers": np.ones(probabilities.shape[1], dtype=float),
        "baseline_score": baseline_score,
        "score": baseline_score,
        "class_recall_deltas": dict.fromkeys(class_labels, 0.0),
        "fold_score_deltas": {int(fold): 0.0 for fold in sorted(np.unique(fold_ids))},
    }


def threshold_stability_results(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
) -> list[dict[str, Any]]:
    """Tune multipliers on 4 folds and evaluate on each held-out fold."""
    results: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold

        fold_multipliers, _train_tuned_score = search_class_multipliers(
            y_true[train_mask],
            probabilities[train_mask],
        )
        baseline_pred = probabilities[valid_mask].argmax(axis=1)
        tuned_pred = predict_with_multipliers(probabilities[valid_mask], fold_multipliers)
        baseline_score = balanced_accuracy(y_true[valid_mask], baseline_pred)
        tuned_score = balanced_accuracy(y_true[valid_mask], tuned_pred)
        before_recall = per_class_recall(y_true[valid_mask], baseline_pred, class_labels)
        after_recall = per_class_recall(y_true[valid_mask], tuned_pred, class_labels)

        results.append(
            {
                "fold": int(fold),
                "multipliers": fold_multipliers.tolist(),
                "baseline_balanced_accuracy": baseline_score,
                "tuned_balanced_accuracy": tuned_score,
                "delta": tuned_score - baseline_score,
                "per_class_recall_before": before_recall,
                "per_class_recall_after": after_recall,
            }
        )
    return results


def _threshold_is_acceptable(
    untuned_score: float,
    tuned_score: float,
    recall_before: dict[str, float],
    recall_after: dict[str, float],
    stability: list[dict[str, Any]],
) -> bool:
    if tuned_score <= untuned_score:
        return False

    class_deltas = [recall_after[label] - recall_before[label] for label in recall_before]
    if min(class_deltas) < -MATERIAL_CLASS_RECALL_REGRESSION:
        return False

    fold_deltas = [result["delta"] for result in stability]
    return min(fold_deltas) >= -MATERIAL_FOLD_REGRESSION


def _print_fold_summary(fold_records: list[dict[str, Any]]) -> None:
    print("fold balanced accuracy:")
    for record in fold_records:
        print(f"  fold {record['fold']}: {record['balanced_accuracy']:.6f}")


def main() -> int:
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    class_labels = encoder.classes_.tolist()

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probabilities = np.zeros((len(X), len(class_labels)), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test), len(class_labels)), dtype=np.float64)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    fold_recalls: list[dict[str, float]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"training fold {fold}/{N_SPLITS}")
        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_columns,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
        )

        valid_probabilities = model.predict_proba(X.iloc[valid_idx])
        oof_probabilities[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test) / N_SPLITS
        fold_ids[valid_idx] = fold

        fold_pred = valid_probabilities.argmax(axis=1)
        fold_score = balanced_accuracy(y[valid_idx], fold_pred)
        fold_recall = per_class_recall(y[valid_idx], fold_pred, class_labels)
        fold_recalls.append(fold_recall)
        fold_records.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_ or MODEL_PARAMS["n_estimators"]),
                "balanced_accuracy": fold_score,
                "per_class_recall": fold_recall,
            }
        )
        print(f"  fold {fold} balanced accuracy: {fold_score:.6f}")

    untuned_pred = oof_probabilities.argmax(axis=1)
    untuned_score = balanced_accuracy(y, untuned_pred)
    untuned_recall = per_class_recall(y, untuned_pred, class_labels)
    multipliers, tuned_score = search_class_multipliers(y, oof_probabilities)
    tuned_pred = predict_with_multipliers(oof_probabilities, multipliers)
    tuned_recall = per_class_recall(y, tuned_pred, class_labels)
    stability = threshold_stability_results(y, oof_probabilities, fold_ids, class_labels)
    unconstrained_threshold_accepted = _threshold_is_acceptable(
        untuned_score,
        tuned_score,
        untuned_recall,
        tuned_recall,
        stability,
    )
    stable_threshold = search_stable_multipliers(y, oof_probabilities, fold_ids, class_labels)
    chosen_multipliers = stable_threshold["multipliers"]
    chosen_pred = predict_with_multipliers(oof_probabilities, chosen_multipliers)
    chosen_score = balanced_accuracy(y, chosen_pred)
    chosen_recall = per_class_recall(y, chosen_pred, class_labels)

    OOF_PROB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OOF_PROB_PATH, oof_probabilities)
    np.save(TEST_PROB_PATH, test_probabilities)

    submission = make_tuned_submission(
        sample_submission["id"],
        test_probabilities,
        chosen_multipliers,
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    recall_warnings = recall_variation_warnings(fold_recalls)
    _print_fold_summary(fold_records)
    print(f"OOF balanced accuracy (argmax): {untuned_score:.6f}")
    print(f"OOF balanced accuracy (tuned candidate): {tuned_score:.6f}")
    print(f"unconstrained threshold accepted: {unconstrained_threshold_accepted}")
    print(f"stable threshold accepted: {stable_threshold['accepted']}")
    print(f"OOF balanced accuracy (chosen): {chosen_score:.6f}")
    print(f"chosen multipliers: {chosen_multipliers.tolist()}")
    if recall_warnings:
        print(f"recall variation warnings: {recall_warnings}")

    record = {
        "params": MODEL_PARAMS,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_columns,
        "folds": fold_records,
        "recall_variation_warnings": recall_warnings,
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "oof_probability_shape": list(oof_probabilities.shape),
        "test_probability_shape": list(test_probabilities.shape),
        "untuned_oof_balanced_accuracy": untuned_score,
        "tuned_candidate_oof_balanced_accuracy": tuned_score,
        "chosen_oof_balanced_accuracy": chosen_score,
        "per_class_recall_before": untuned_recall,
        "per_class_recall_candidate_after": tuned_recall,
        "per_class_recall_chosen_after": chosen_recall,
        "candidate_multipliers": multipliers.tolist(),
        "chosen_multipliers": chosen_multipliers.tolist(),
        "unconstrained_threshold_accepted": unconstrained_threshold_accepted,
        "stable_threshold": {
            **stable_threshold,
            "multipliers": stable_threshold["multipliers"].tolist(),
        },
        "threshold_stability": stability,
        "submission_path": str(SUBMISSION_PATH),
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    write_json(EXPERIMENT_PATH, record)
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    print(f"wrote {OOF_PROB_PATH}")
    print(f"wrote {TEST_PROB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
