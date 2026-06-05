"""Leakage-safe target-encoding experiment for soft categorical/redshift groups."""
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
N_CLASSES = 3
SMOOTHING = 20.0

RUNS_PATH = Path("experiments/runs.jsonl")
EXPERIMENT_PATH = Path("experiments/10_target_encoding.json")
OOF_PROB_PATH = Path("experiments/10_target_encoding_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/10_target_encoding_test_probabilities.npy")
SUBMISSION_PATH = Path("submissions/10_target_encoding.csv")

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

GROUP_COLUMNS = [
    "spectral_type",
    "galaxy_population",
    "spectral_population",
    "redshift_bin",
    "spectral_population_redshift_bin",
]


def build_redshift_bins(redshift: pd.Series) -> pd.Series:
    """Return coarse redshift bins as a categorical series."""
    return pd.cut(
        redshift,
        bins=[-np.inf, 0.02, 0.08, 0.15, 0.35, 0.75, 1.5, np.inf],
        labels=[
            "very_low",
            "star_core",
            "star_edge",
            "low_galaxy",
            "mid_galaxy",
            "qso_overlap",
            "high_qso",
        ],
    ).astype("category")


def fit_apply_target_encoding(
    train_categories: pd.Series,
    y: np.ndarray,
    apply_categories: pd.Series,
    n_classes: int = N_CLASSES,
    smoothing: float = SMOOTHING,
) -> np.ndarray:
    """Fit smoothed class-rate encodings and apply them to another category series."""
    train_key = train_categories.astype("string")
    apply_key = apply_categories.astype("string")
    global_prior = np.bincount(y, minlength=n_classes).astype(float)
    global_prior /= global_prior.sum()

    frame = pd.DataFrame({"category": train_key, "target": y})
    counts = frame.groupby("category", observed=True).size()
    encoded_by_category: dict[str, np.ndarray] = {}
    for category, count in counts.items():
        category_targets = frame.loc[frame["category"] == category, "target"].to_numpy()
        class_counts = np.bincount(category_targets, minlength=n_classes).astype(float)
        encoded_by_category[str(category)] = (class_counts + (global_prior * smoothing)) / (
            count + smoothing
        )

    output = np.vstack(
        [encoded_by_category.get(str(category), global_prior) for category in apply_key]
    )
    return output


def add_group_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Add group columns used only to fit target encodings."""
    enriched = X.copy()
    enriched["redshift_bin"] = build_redshift_bins(enriched["redshift"])
    enriched["spectral_population_redshift_bin"] = (
        enriched["spectral_population"].astype("string")
        + "__"
        + enriched["redshift_bin"].astype("string")
    )
    return enriched


def add_target_encoding_columns(
    X_train_base: pd.DataFrame,
    X_apply_base: pd.DataFrame,
    y_train: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    """Return ``X_apply_base`` plus target-encoding columns fit from ``X_train_base``."""
    train_groups = add_group_columns(X_train_base)
    apply_groups = add_group_columns(X_apply_base)
    output = X_apply_base.copy()
    for column in GROUP_COLUMNS:
        encoded = fit_apply_target_encoding(train_groups[column], y_train, apply_groups[column])
        for class_idx in range(N_CLASSES):
            output[f"{prefix}_{column}_class_{class_idx}"] = encoded[:, class_idx]
    return output


def make_target_encoding_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def run_target_encoding_cv() -> dict[str, Any]:
    """Evaluate target encodings with OOF-safe train encodings."""
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probabilities = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    folds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"target encoding fold {fold}/{N_SPLITS}")
        X_train_fold = add_target_encoding_columns(
            X.iloc[train_idx],
            X.iloc[train_idx],
            y[train_idx],
            prefix="te",
        )
        X_valid_fold = add_target_encoding_columns(
            X.iloc[train_idx],
            X.iloc[valid_idx],
            y[train_idx],
            prefix="te",
        )
        X_test_fold = add_target_encoding_columns(
            X.iloc[train_idx],
            X_test,
            y[train_idx],
            prefix="te",
        )
        model = LGBMClassifier(**BASE_PARAMS, random_state=SEED + fold)
        model.fit(
            X_train_fold,
            y[train_idx],
            eval_set=[(X_valid_fold, y[valid_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_columns,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
        )
        valid_probabilities = model.predict_proba(X_valid_fold)
        oof_probabilities[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test_fold) / N_SPLITS
        fold_ids[valid_idx] = fold
        folds.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_ or BASE_PARAMS["n_estimators"]),
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_probabilities.argmax(axis=1)),
                "per_class_recall": per_class_recall(y[valid_idx], valid_probabilities.argmax(axis=1), class_labels),
            }
        )

    # Reuse the stable threshold helper from Phase 4 to avoid aggressive continuous shifts.
    import importlib.util

    spec = importlib.util.spec_from_file_location("tune_script", Path("scripts/03_tune.py"))
    tune = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tune)
    threshold = tune.search_stable_multipliers(y, oof_probabilities, fold_ids, class_labels)
    chosen_pred = predict_with_multipliers(oof_probabilities, threshold["multipliers"])
    chosen_score = balanced_accuracy(y, chosen_pred)

    np.save(OOF_PROB_PATH, oof_probabilities)
    np.save(TEST_PROB_PATH, test_probabilities)
    submission = make_target_encoding_submission(
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
        "candidate": "target_encoding",
        "reference_score": REFERENCE_SCORE,
        "argmax_oof_balanced_accuracy": balanced_accuracy(y, oof_probabilities.argmax(axis=1)),
        "chosen_oof_balanced_accuracy": chosen_score,
        "delta_vs_reference": chosen_score - REFERENCE_SCORE,
        "chosen_multipliers": threshold["multipliers"].tolist(),
        "per_class_recall": per_class_recall(y, chosen_pred, class_labels),
        "folds": folds,
        "group_columns": GROUP_COLUMNS,
        "smoothing": SMOOTHING,
        "params": BASE_PARAMS,
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "target_encoding", **record})
    return record


def main() -> int:
    record = run_target_encoding_cv()
    print(f"target encoding argmax OOF: {record['argmax_oof_balanced_accuracy']:.6f}")
    print(f"target encoding tuned OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs reference: {record['delta_vs_reference']:.6f}")
    print(f"multipliers: {record['chosen_multipliers']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
