"""Generate the best constrained blend using target-encoding diversity."""
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
TARGET_ENCODING_OOF_PATH = Path("experiments/10_target_encoding_oof_probabilities.npy")
TARGET_ENCODING_TEST_PATH = Path("experiments/10_target_encoding_test_probabilities.npy")

EXPERIMENT_PATH = Path("experiments/11_target_encoding_blend.json")
SUBMISSION_PATH = Path("submissions/11_target_encoding_blend.csv")
RUNS_PATH = Path("experiments/runs.jsonl")

WEIGHTS = np.array([0.5, 0.4, 0.1])
MULTIPLIERS = np.array([0.8, 1.1, 1.15])
REFERENCE_MULTIPLIERS = np.array([0.9, 0.8, 1.15])


def target_encoding_blend(
    reference_probabilities: np.ndarray,
    xgboost_probabilities: np.ndarray,
    target_encoding_probabilities: np.ndarray,
) -> np.ndarray:
    """Blend reference, XGBoost, and target-encoding probabilities."""
    if not (
        reference_probabilities.shape
        == xgboost_probabilities.shape
        == target_encoding_probabilities.shape
    ):
        raise ValueError("probability arrays must have matching shapes")
    return (
        (reference_probabilities * WEIGHTS[0])
        + (xgboost_probabilities * WEIGHTS[1])
        + (target_encoding_probabilities * WEIGHTS[2])
    )


def make_target_encoding_blend_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def run_target_encoding_blend() -> dict[str, object]:
    """Write target-encoding blend submission and experiment record."""
    train_df, _test_df, sample_submission = load_raw()
    _X, y, _categorical_columns, encoder = build_features(train_df)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()

    reference_oof = np.load(REFERENCE_OOF_PATH)
    reference_test = np.load(REFERENCE_TEST_PATH)
    xgboost_oof = np.load(XGBOOST_OOF_PATH)
    xgboost_test = np.load(XGBOOST_TEST_PATH)
    target_encoding_oof = np.load(TARGET_ENCODING_OOF_PATH)
    target_encoding_test = np.load(TARGET_ENCODING_TEST_PATH)

    reference_pred = predict_with_multipliers(reference_oof, REFERENCE_MULTIPLIERS)
    reference_score = balanced_accuracy(y, reference_pred)
    reference_recall = per_class_recall(y, reference_pred, class_labels)

    blended_oof = target_encoding_blend(reference_oof, xgboost_oof, target_encoding_oof)
    blended_test = target_encoding_blend(reference_test, xgboost_test, target_encoding_test)
    blended_pred = predict_with_multipliers(blended_oof, MULTIPLIERS)
    blended_score = balanced_accuracy(y, blended_pred)
    blended_recall = per_class_recall(y, blended_pred, class_labels)
    recall_deltas = {
        label: blended_recall[label] - reference_recall[label] for label in class_labels
    }

    submission = make_target_encoding_blend_submission(
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
        "candidate": "target_encoding_blend",
        "weights": {
            "lgbm_seed_average_final": float(WEIGHTS[0]),
            "xgboost": float(WEIGHTS[1]),
            "target_encoding": float(WEIGHTS[2]),
        },
        "multipliers": MULTIPLIERS.tolist(),
        "reference_oof_balanced_accuracy": reference_score,
        "chosen_oof_balanced_accuracy": blended_score,
        "delta_vs_reference": blended_score - reference_score,
        "reference_per_class_recall": reference_recall,
        "per_class_recall": blended_recall,
        "per_class_recall_delta": recall_deltas,
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "target_encoding_blend", **record})
    return record


def main() -> int:
    record = run_target_encoding_blend()
    print(f"target-encoding blend OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs 03_final OOF: {record['delta_vs_reference']:.6f}")
    print(f"recall deltas: {record['per_class_recall_delta']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
