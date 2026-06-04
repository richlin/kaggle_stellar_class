"""Train the Phase 1 LightGBM baseline and write a scoreable submission."""
# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import build_features, load_raw
from src.validate import validate_submission

SEED = 42
SUBMISSION_PATH = Path("submissions/01_baseline.csv")
EXPERIMENT_PATH = Path("experiments/01_baseline.json")
MODEL_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def make_submission(ids: pd.Series, encoded_predictions: np.ndarray, encoder: LabelEncoder) -> pd.DataFrame:
    """Create a submission DataFrame from encoded model predictions."""
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": ids.to_numpy(), "class": labels})


def write_experiment_record(path: str | Path, record: dict[str, Any]) -> None:
    """Write a JSON experiment record."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, encoder: LabelEncoder) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=np.arange(len(encoder.classes_)), average=None)
    return {label: float(score) for label, score in zip(encoder.classes_, recalls, strict=True)}


def main() -> int:
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train, categorical_feature=categorical_columns)
    holdout_pred = model.predict(X_holdout)

    holdout_score = balanced_accuracy_score(y_holdout, holdout_pred)
    per_class_recall = _per_class_recall(y_holdout, holdout_pred, encoder)
    matrix = confusion_matrix(y_holdout, holdout_pred, labels=np.arange(len(encoder.classes_)))

    print(f"holdout balanced accuracy: {holdout_score:.6f}")
    print("per-class recall:")
    for label, score in per_class_recall.items():
        print(f"  {label}: {score:.6f}")
    print("confusion matrix:")
    print(matrix)

    final_model = LGBMClassifier(**MODEL_PARAMS)
    final_model.fit(X, y, categorical_feature=categorical_columns)
    test_pred = final_model.predict(X_test)

    submission = make_submission(sample_submission["id"], test_pred, encoder)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record = {
        "params": MODEL_PARAMS,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_columns,
        "seed": SEED,
        "holdout_balanced_accuracy": float(holdout_score),
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix.tolist(),
        "submission_path": str(SUBMISSION_PATH),
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    write_experiment_record(EXPERIMENT_PATH, record)
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
