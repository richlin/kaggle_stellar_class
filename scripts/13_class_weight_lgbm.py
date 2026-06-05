"""Train a class-adjusted LightGBM candidate and blend it with the current best."""
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
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
from sklearn.utils.class_weight import compute_sample_weight

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
CURRENT_BEST_SCORE = 0.9662824834818386
N_SPLITS = 5
SEED = 62
N_CLASSES = 3
CLASS_ADJUSTMENT = np.array([1.0, 1.1, 1.05])
REFERENCE_MULTIPLIERS = np.array([0.9, 0.8, 1.15])

RUNS_PATH = Path("experiments/runs.jsonl")
EXPERIMENT_PATH = Path("experiments/13_class_weight_lgbm.json")
OOF_PROB_PATH = Path("experiments/13_class_weight_lgbm_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/13_class_weight_lgbm_test_probabilities.npy")
SUBMISSION_PATH = Path("submissions/13_class_weight_lgbm.csv")

CURRENT_BEST_OOF_PATH = Path("experiments/12_multi_blend_oof_probabilities.npy")
CURRENT_BEST_TEST_PATH = Path("experiments/12_multi_blend_test_probabilities.npy")

BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "n_estimators": 900,
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


def _load_ensemble_module():
    spec = importlib.util.spec_from_file_location("ensemble_script", Path("scripts/04_ensemble.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def class_adjusted_sample_weights(
    y: np.ndarray,
    class_adjustment: np.ndarray = CLASS_ADJUSTMENT,
) -> np.ndarray:
    """Return balanced sample weights multiplied by class-level adjustments."""
    if class_adjustment.shape != (N_CLASSES,):
        raise ValueError(f"class_adjustment shape must be {(N_CLASSES,)}, got {class_adjustment.shape}")
    if np.any(class_adjustment <= 0):
        raise ValueError("class adjustments must be positive")

    weights = compute_sample_weight("balanced", y).astype(float)
    weights *= class_adjustment[y]
    return weights / weights.mean()


def make_class_weight_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def run_class_weight_lgbm() -> dict[str, Any]:
    """Train class-adjusted LightGBM probabilities and write the best local blend."""
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()

    sample_weights = class_adjusted_sample_weights(y)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probabilities = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    folds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"class-adjusted lgbm fold {fold}/{N_SPLITS}")
        model = LGBMClassifier(**BASE_PARAMS, random_state=SEED + fold)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            sample_weight=sample_weights[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_columns,
            callbacks=[early_stopping(stopping_rounds=60, verbose=False), log_evaluation(period=0)],
        )
        valid_probabilities = model.predict_proba(X.iloc[valid_idx])
        oof_probabilities[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test) / N_SPLITS
        fold_ids[valid_idx] = fold
        folds.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_ or BASE_PARAMS["n_estimators"]),
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_probabilities.argmax(axis=1)),
            }
        )

    np.save(OOF_PROB_PATH, oof_probabilities)
    np.save(TEST_PROB_PATH, test_probabilities)

    ensemble = _load_ensemble_module()
    reference_oof = np.load("experiments/03_final_oof_probabilities.npy")
    reference_pred = predict_with_multipliers(reference_oof, REFERENCE_MULTIPLIERS)
    reference_recall = per_class_recall(y, reference_pred, class_labels)

    standalone_threshold = ensemble.search_continuous_multipliers(
        y,
        oof_probabilities,
        fold_ids,
        class_labels,
    )
    standalone_pred = predict_with_multipliers(oof_probabilities, standalone_threshold["multipliers"])

    current_best_oof = np.load(CURRENT_BEST_OOF_PATH)
    current_best_test = np.load(CURRENT_BEST_TEST_PATH)
    blend = ensemble.search_blend_weights(
        y,
        [current_best_oof, oof_probabilities],
        fold_ids,
        class_labels,
    )
    blended_oof = ensemble.weighted_probability_blend(
        [current_best_oof, oof_probabilities],
        blend["weights"],
    )
    blended_test = ensemble.weighted_probability_blend(
        [current_best_test, test_probabilities],
        blend["weights"],
    )
    blended_pred = predict_with_multipliers(blended_oof, blend["multipliers"])
    blended_score = balanced_accuracy(y, blended_pred)
    blended_recall = per_class_recall(y, blended_pred, class_labels)

    submission = make_class_weight_submission(
        sample_submission,
        blended_test,
        blend["multipliers"],
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate": "class_weight_lgbm",
        "reference_score": REFERENCE_SCORE,
        "current_best_score": CURRENT_BEST_SCORE,
        "class_adjustment": CLASS_ADJUSTMENT.tolist(),
        "params": BASE_PARAMS,
        "folds": folds,
        "argmax_oof_balanced_accuracy": balanced_accuracy(y, oof_probabilities.argmax(axis=1)),
        "standalone_oof_balanced_accuracy": standalone_threshold["score"],
        "standalone_multipliers": standalone_threshold["multipliers"].tolist(),
        "standalone_per_class_recall": per_class_recall(y, standalone_pred, class_labels),
        "blend_weights": {
            "multi_blend_12": float(blend["weights"][0]),
            "class_weight_lgbm": float(blend["weights"][1]),
        },
        "blend_multipliers": blend["multipliers"].tolist(),
        "chosen_oof_balanced_accuracy": blended_score,
        "delta_vs_current_best": blended_score - CURRENT_BEST_SCORE,
        "reference_per_class_recall": reference_recall,
        "per_class_recall": blended_recall,
        "per_class_recall_delta_vs_reference": {
            label: blended_recall[label] - reference_recall[label] for label in class_labels
        },
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "class_weight_lgbm", **record})
    return record


def main() -> int:
    record = run_class_weight_lgbm()
    print(f"class-weight lgbm argmax OOF: {record['argmax_oof_balanced_accuracy']:.6f}")
    print(f"class-weight lgbm standalone OOF: {record['standalone_oof_balanced_accuracy']:.6f}")
    print(f"class-weight lgbm blend OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs 12_multi_blend: {record['delta_vs_current_best']:.6f}")
    print(f"blend weights: {record['blend_weights']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
