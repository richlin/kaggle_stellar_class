"""Cross-validated stacker over saved base-model probability artifacts."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
STABLE_THRESHOLD_GRID = np.array([0.75, 0.8, 0.9, 1.0, 1.1, 1.15])
MATERIAL_CLASS_RECALL_REGRESSION = 0.003
MATERIAL_FOLD_REGRESSION = 0.002

EXPERIMENT_PATH = Path("experiments/07_probability_stacker.json")
RUNS_PATH = Path("experiments/runs.jsonl")
SUBMISSION_PATH = Path("submissions/07_probability_stacker.csv")

BASE_PROBABILITY_SETS = [
    (
        "lgbm_seed_average_final",
        Path("experiments/03_final_oof_probabilities.npy"),
        Path("experiments/03_final_test_probabilities.npy"),
    ),
    ("xgboost", Path("experiments/04_xgboost_oof_probabilities.npy"), Path("experiments/04_xgboost_test_probabilities.npy")),
    ("catboost", Path("experiments/04_catboost_oof_probabilities.npy"), Path("experiments/04_catboost_test_probabilities.npy")),
    ("lgbm_dart", Path("experiments/04_lgbm_dart_oof_probabilities.npy"), Path("experiments/04_lgbm_dart_test_probabilities.npy")),
    ("boundary_v1", Path("experiments/05_boundary_v1_oof_probabilities.npy"), Path("experiments/05_boundary_v1_test_probabilities.npy")),
]


def build_stack_features(probability_sets: list[np.ndarray]) -> np.ndarray:
    """Build row-level meta-features from base-model probability matrices."""
    if not probability_sets:
        raise ValueError("at least one probability matrix is required")
    first_shape = probability_sets[0].shape
    for probabilities in probability_sets:
        if probabilities.shape != first_shape:
            raise ValueError("all probability matrices must have matching shape")

    blocks = [*probability_sets]
    for probabilities in probability_sets:
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
        blocks.append(margin.reshape(-1, 1))
        blocks.append(entropy.reshape(-1, 1))
    return np.hstack(blocks)


def select_stacker_candidate(candidate_score: float, reference_score: float) -> dict[str, Any]:
    """Accept only stackers that beat the public-best reference OOF."""
    return {
        "accepted": candidate_score > reference_score,
        "candidate_score": float(candidate_score),
        "reference_score": float(reference_score),
        "delta": float(candidate_score - reference_score),
    }


def make_stacker_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def _load_probability_sets() -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    names = []
    oof_sets = []
    test_sets = []
    for name, oof_path, test_path in BASE_PROBABILITY_SETS:
        if oof_path.exists() and test_path.exists():
            names.append(name)
            oof_sets.append(np.load(oof_path))
            test_sets.append(np.load(test_path))
    if not names:
        raise FileNotFoundError("no saved probability artifacts found")
    return names, oof_sets, test_sets


def _fold_scores(y_true: np.ndarray, y_pred: np.ndarray, fold_ids: np.ndarray) -> dict[int, float]:
    return {
        int(fold): balanced_accuracy(y_true[fold_ids == fold], y_pred[fold_ids == fold])
        for fold in sorted(np.unique(fold_ids))
    }


def search_stable_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    """Search stable-grid class multipliers for stacker probabilities."""
    baseline_pred = probabilities.argmax(axis=1)
    baseline_score = balanced_accuracy(y_true, baseline_pred)
    baseline_recall = per_class_recall(y_true, baseline_pred, class_labels)
    baseline_fold_scores = _fold_scores(y_true, baseline_pred, fold_ids)
    best = {
        "accepted": False,
        "multipliers": np.ones(probabilities.shape[1], dtype=float),
        "baseline_score": baseline_score,
        "score": baseline_score,
        "class_recall_deltas": dict.fromkeys(class_labels, 0.0),
        "fold_score_deltas": dict.fromkeys(baseline_fold_scores, 0.0),
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

        fold_scores = _fold_scores(y_true, pred, fold_ids)
        fold_deltas = {fold: fold_scores[fold] - baseline_fold_scores[fold] for fold in fold_scores}
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


def run_stacker() -> dict[str, Any]:
    """Train a cross-validated logistic stacker from saved OOF probabilities."""
    train_df, _test_df, sample_submission = load_raw()
    _X, y, _categorical_columns, encoder = build_features(train_df)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()

    names, oof_sets, test_sets = _load_probability_sets()
    X_stack = build_stack_features(oof_sets)
    X_test_stack = build_stack_features(test_sets)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probabilities = np.zeros((len(X_stack), len(class_labels)), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test_stack), len(class_labels)), dtype=np.float64)
    fold_ids = np.full(len(X_stack), -1, dtype=np.int16)
    fold_records = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_stack, y), start=1):
        print(f"stacker fold {fold}/{N_SPLITS}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.6,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED + fold,
            ),
        )
        model.fit(X_stack[train_idx], y[train_idx])
        valid_probabilities = model.predict_proba(X_stack[valid_idx])
        oof_probabilities[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test_stack) / N_SPLITS
        fold_ids[valid_idx] = fold
        fold_records.append(
            {
                "fold": fold,
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_probabilities.argmax(axis=1)),
                "per_class_recall": per_class_recall(y[valid_idx], valid_probabilities.argmax(axis=1), class_labels),
            }
        )

    threshold = search_stable_multipliers(y, oof_probabilities, fold_ids, class_labels)
    chosen_pred = predict_with_multipliers(oof_probabilities, threshold["multipliers"])
    chosen_score = balanced_accuracy(y, chosen_pred)
    selection = select_stacker_candidate(chosen_score, REFERENCE_SCORE)
    submission = make_stacker_submission(
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
        "candidate": "probability_stacker",
        "base_probability_sets": names,
        "stack_feature_shape": list(X_stack.shape),
        "argmax_oof_balanced_accuracy": balanced_accuracy(y, oof_probabilities.argmax(axis=1)),
        "chosen_oof_balanced_accuracy": chosen_score,
        "candidate_selection": selection,
        "chosen_multipliers": threshold["multipliers"].tolist(),
        "per_class_recall": per_class_recall(y, chosen_pred, class_labels),
        "stable_threshold": {
            **threshold,
            "multipliers": threshold["multipliers"].tolist(),
        },
        "folds": fold_records,
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "probability_stacker", **record})
    return record


def main() -> int:
    record = run_stacker()
    print(f"stacker argmax OOF: {record['argmax_oof_balanced_accuracy']:.6f}")
    print(f"stacker tuned OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"accepted: {record['candidate_selection']['accepted']}")
    print(f"multipliers: {record['chosen_multipliers']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
