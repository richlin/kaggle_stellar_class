"""Generate a constrained multi-model blend candidate from cached probabilities."""
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

REFERENCE_OOF_PATH = Path("experiments/03_final_oof_probabilities.npy")
REFERENCE_TEST_PATH = Path("experiments/03_final_test_probabilities.npy")
XGBOOST_OOF_PATH = Path("experiments/04_xgboost_oof_probabilities.npy")
XGBOOST_TEST_PATH = Path("experiments/04_xgboost_test_probabilities.npy")
EXTENDED_OOF_PATH = Path("experiments/09_extended_seed_average_oof_probabilities.npy")
EXTENDED_TEST_PATH = Path("experiments/09_extended_seed_average_test_probabilities.npy")
BOUNDARY_OOF_PATH = Path("experiments/05_boundary_v1_oof_probabilities.npy")
BOUNDARY_TEST_PATH = Path("experiments/05_boundary_v1_test_probabilities.npy")

EXPERIMENT_PATH = Path("experiments/12_multi_blend.json")
SUBMISSION_PATH = Path("submissions/12_multi_blend.csv")
OOF_PROB_PATH = Path("experiments/12_multi_blend_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/12_multi_blend_test_probabilities.npy")
RUNS_PATH = Path("experiments/runs.jsonl")

MODEL_NAMES = [
    "lgbm_seed_average_final",
    "xgboost",
    "extended_seed_average",
    "boundary_v1",
]
WEIGHTS = np.array([0.23, 0.44, 0.28, 0.05])
MULTIPLIERS = np.array([0.74, 0.94, 1.05])
REFERENCE_MULTIPLIERS = np.array([0.9, 0.8, 1.15])


def multi_model_blend(probabilities: list[np.ndarray]) -> np.ndarray:
    """Blend the four cached probability arrays with fixed normalized weights."""
    if len(probabilities) != len(WEIGHTS):
        raise ValueError(f"expected exactly {len(WEIGHTS)} probability arrays")

    first_shape = probabilities[0].shape
    for probability in probabilities:
        if probability.shape != first_shape:
            raise ValueError("all probability arrays must have matching shapes")

    normalized_weights = WEIGHTS / WEIGHTS.sum()
    blended = np.zeros_like(probabilities[0], dtype=float)
    for probability, weight in zip(probabilities, normalized_weights, strict=True):
        blended += probability * weight
    return blended


def make_multi_blend_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def run_multi_blend() -> dict[str, object]:
    """Write the multi-blend submission and experiment record."""
    train_df, _test_df, sample_submission = load_raw()
    _X, y, _categorical_columns, encoder = build_features(train_df)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()

    reference_oof = np.load(REFERENCE_OOF_PATH)
    reference_test = np.load(REFERENCE_TEST_PATH)
    xgboost_oof = np.load(XGBOOST_OOF_PATH)
    xgboost_test = np.load(XGBOOST_TEST_PATH)
    extended_oof = np.load(EXTENDED_OOF_PATH)
    extended_test = np.load(EXTENDED_TEST_PATH)
    boundary_oof = np.load(BOUNDARY_OOF_PATH)
    boundary_test = np.load(BOUNDARY_TEST_PATH)

    reference_pred = predict_with_multipliers(reference_oof, REFERENCE_MULTIPLIERS)
    reference_score = balanced_accuracy(y, reference_pred)
    reference_recall = per_class_recall(y, reference_pred, class_labels)

    blended_oof = multi_model_blend([reference_oof, xgboost_oof, extended_oof, boundary_oof])
    blended_test = multi_model_blend([reference_test, xgboost_test, extended_test, boundary_test])
    blended_pred = predict_with_multipliers(blended_oof, MULTIPLIERS)
    blended_score = balanced_accuracy(y, blended_pred)
    blended_recall = per_class_recall(y, blended_pred, class_labels)
    recall_deltas = {
        label: blended_recall[label] - reference_recall[label] for label in class_labels
    }

    OOF_PROB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OOF_PROB_PATH, blended_oof)
    np.save(TEST_PROB_PATH, blended_test)

    submission = make_multi_blend_submission(
        sample_submission,
        blended_test,
        MULTIPLIERS,
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate": "multi_blend",
        "weights": {
            name: float(weight) for name, weight in zip(MODEL_NAMES, WEIGHTS, strict=True)
        },
        "multipliers": MULTIPLIERS.tolist(),
        "reference_oof_balanced_accuracy": reference_score,
        "chosen_oof_balanced_accuracy": blended_score,
        "delta_vs_reference": blended_score - reference_score,
        "reference_per_class_recall": reference_recall,
        "per_class_recall": blended_recall,
        "per_class_recall_delta": recall_deltas,
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "multi_blend", **record})
    return record


def main() -> int:
    record = run_multi_blend()
    print(f"multi-blend OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs 03_final OOF: {record['delta_vs_reference']:.6f}")
    print(f"recall deltas: {record['per_class_recall_delta']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
