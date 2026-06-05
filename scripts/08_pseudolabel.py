"""Generate a low-weight high-confidence pseudo-label submission."""
# ruff: noqa: E402
from __future__ import annotations

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
from sklearn.preprocessing import LabelEncoder

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import append_jsonl, write_json

EXPERIMENT_PATH = Path("experiments/08_pseudolabel.json")
RUNS_PATH = Path("experiments/runs.jsonl")
SUBMISSION_PATH = Path("submissions/08_pseudolabel.csv")

REFERENCE_TEST_PATH = Path("experiments/03_final_test_probabilities.npy")
XGBOOST_TEST_PATH = Path("experiments/04_xgboost_test_probabilities.npy")

STAR_SAFE_MULTIPLIERS = np.array([0.8, 1.1, 1.15])
PSEUDOLABEL_WEIGHT = 0.15
MIN_PROBABILITY = 0.995
MIN_MARGIN = 0.75
SEEDS = [42, 43, 44]

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


def select_pseudolabels(
    probabilities: np.ndarray,
    min_probability: float = MIN_PROBABILITY,
    min_margin: float = MIN_MARGIN,
) -> dict[str, np.ndarray]:
    """Select high-confidence pseudo labels from normalized probabilities."""
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    sorted_probabilities = np.sort(normalized, axis=1)
    max_probability = sorted_probabilities[:, -1]
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    mask = (max_probability >= min_probability) & (margin >= min_margin)
    return {"mask": mask, "labels": normalized[mask].argmax(axis=1)}


def make_pseudolabel_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    labels = encoder.inverse_transform(probabilities.argmax(axis=1))
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def _star_safe_test_probabilities() -> np.ndarray:
    reference_test = np.load(REFERENCE_TEST_PATH)
    xgboost_test = np.load(XGBOOST_TEST_PATH)
    blended = (reference_test * 0.6) + (xgboost_test * 0.4)
    adjusted = blended * STAR_SAFE_MULTIPLIERS
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def run_pseudolabel() -> dict[str, Any]:
    """Train final models with low-weight high-confidence pseudo-label rows."""
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    if y is None:
        raise ValueError("training labels are required")

    pseudo_source = _star_safe_test_probabilities()
    selected = select_pseudolabels(pseudo_source)
    pseudo_X = X_test.iloc[selected["mask"]]
    pseudo_y = selected["labels"]

    X_augmented = pd.concat([X, pseudo_X], axis=0, ignore_index=True)
    y_augmented = np.concatenate([y, pseudo_y])
    sample_weight = np.concatenate(
        [
            np.ones(len(y), dtype=float),
            np.full(len(pseudo_y), PSEUDOLABEL_WEIGHT, dtype=float),
        ]
    )

    test_probabilities = np.zeros((len(X_test), len(encoder.classes_)), dtype=np.float64)
    for seed in SEEDS:
        print(f"pseudo-label final model seed {seed}")
        model = LGBMClassifier(**BASE_PARAMS, random_state=seed)
        model.fit(
            X_augmented,
            y_augmented,
            sample_weight=sample_weight,
            categorical_feature=categorical_columns,
        )
        test_probabilities += model.predict_proba(X_test) / len(SEEDS)

    submission = make_pseudolabel_submission(sample_submission, test_probabilities, encoder)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    pseudo_counts = pd.Series(encoder.inverse_transform(pseudo_y)).value_counts().sort_index().to_dict()
    submission_counts = submission["class"].value_counts().sort_index().to_dict()
    record = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate": "low_weight_pseudolabel",
        "source": "star_safe_blend_test_probabilities",
        "min_probability": MIN_PROBABILITY,
        "min_margin": MIN_MARGIN,
        "pseudolabel_weight": PSEUDOLABEL_WEIGHT,
        "seeds": SEEDS,
        "pseudo_label_count": int(len(pseudo_y)),
        "pseudo_label_counts": pseudo_counts,
        "submission_counts": submission_counts,
        "params": BASE_PARAMS,
        "submission_path": str(SUBMISSION_PATH),
        "risk_note": (
            "Transductive pseudo-label candidate; no honest OOF score because selected rows come "
            "from the test set. Keep 03_final as fallback unless public score improves."
        ),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "pseudolabel", **record})
    return record


def main() -> int:
    record = run_pseudolabel()
    print(f"pseudo labels: {record['pseudo_label_count']}")
    print(f"pseudo label counts: {record['pseudo_label_counts']}")
    print(f"submission counts: {record['submission_counts']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
