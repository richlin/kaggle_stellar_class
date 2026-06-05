"""Task 24 follow-up: full-data LOO spatial final candidate.

The public top-10 cluster suggests the final test-time spatial signal matters
more than another residual classifier. The existing `15/16` models train on
KFold-OOF spatial features, while test rows use all train labels as neighbours.
This script trains the LightGBM component on leave-one-out spatial features so
train-time features have the same density as test-time features.
"""
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
from lightgbm import LGBMClassifier

from src.data import build_features, load_raw
from src.spatial import loo_neighbour_features, radec_to_xyz
from src.validate import validate_submission
from src.validation import write_json

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
KS = [5, 10, 25, 50, 100, 250]
MAX_K = 250
SMOOTHING = 10.0
SEEDS = [42, 43, 44]
SPATIAL_BLEND_WEIGHT_LGBM = 0.55
MULTIPLIERS = np.array([0.45, 0.75, 1.0])

PARAMS = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 900,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
}

LOO_TRAIN = PROJECT_ROOT / "experiments" / "19_loo_spatial_train_features.npy"
LOO_NAMES = PROJECT_ROOT / "experiments" / "19_loo_spatial_feature_names.npy"
TEST_SPATIAL = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
LOO_TEST_PROB = PROJECT_ROOT / "experiments" / "19_loo_spatial_lgbm_test_probabilities.npy"
BLEND_TEST_PROB = PROJECT_ROOT / "experiments" / "19_loo_spatial_blend_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "19_loo_spatial_final.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "19_loo_spatial_final.csv"


def build_loo_spatial(train: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build or load leave-one-out train spatial features."""
    if LOO_TRAIN.exists() and LOO_NAMES.exists():
        return np.load(LOO_TRAIN), list(np.load(LOO_NAMES, allow_pickle=True))

    xyz = radec_to_xyz(train["alpha"].to_numpy(), train["delta"].to_numpy())
    priors = np.bincount(y, minlength=len(CLASS_LABELS)) / len(y)
    features, names = loo_neighbour_features(
        xyz,
        y,
        KS,
        n_classes=len(CLASS_LABELS),
        priors=priors,
        smoothing=SMOOTHING,
        max_k=MAX_K,
    )
    np.save(LOO_TRAIN, features)
    np.save(LOO_NAMES, np.array(names, dtype=object))
    return features, names


def add_spatial_features(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    train_features: np.ndarray,
    test_features: np.ndarray,
    names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = pd.DataFrame(train_features, columns=names, index=X.index)
    test_frame = pd.DataFrame(test_features, columns=names, index=X_test.index)
    return pd.concat([X, train_frame], axis=1), pd.concat([X_test, test_frame], axis=1)


def train_full_lgbm(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Train full-data seeded LightGBM models and average test probabilities."""
    test_prob = np.zeros((len(X_test), len(CLASS_LABELS)), dtype=float)
    for seed in SEEDS:
        print(f"  full LOO LightGBM seed {seed}")
        model = LGBMClassifier(**PARAMS, random_state=seed)
        model.fit(X, y, categorical_feature=cat_cols)
        test_prob += model.predict_proba(X_test) / len(SEEDS)
    return test_prob


def main() -> int:
    train, test, sample = load_raw()
    X, y, cat_cols, encoder = build_features(train)
    if y is None:
        raise ValueError("training data must include class labels")
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    loo_features, names = build_loo_spatial(train, y)
    X, X_test = add_spatial_features(X, X_test, loo_features, np.load(TEST_SPATIAL), names)
    loo_lgbm_test = train_full_lgbm(X, y, X_test, cat_cols)
    np.save(LOO_TEST_PROB, loo_lgbm_test)

    blend_test = (
        SPATIAL_BLEND_WEIGHT_LGBM * loo_lgbm_test
        + (1 - SPATIAL_BLEND_WEIGHT_LGBM) * np.load(XGB_TEST)
    )
    np.save(BLEND_TEST_PROB, blend_test)
    predicted = (blend_test * MULTIPLIERS).argmax(axis=1)
    submission = pd.DataFrame(
        {
            "id": sample["id"].to_numpy(),
            "class": encoder.inverse_transform(predicted),
        }
    )
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)

    write_json(
        EXPERIMENT,
        {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
            "rationale": (
                "Train full-data LightGBM on leave-one-out spatial features to reduce "
                "train/test spatial feature mismatch; no honest OOF score is available "
                "for this final-only candidate."
            ),
            "seeds": SEEDS,
            "ks": KS,
            "smoothing": SMOOTHING,
            "params": PARAMS,
            "blend_weight_lgbm": SPATIAL_BLEND_WEIGHT_LGBM,
            "multipliers": MULTIPLIERS.tolist(),
            "submission_path": str(SUBMISSION),
        },
    )
    print("\n================ LOO SPATIAL FINAL CANDIDATE ================")
    print("honest OOF          : n/a (final-only train/test mismatch candidate)")
    print(f"blend weight LGBM   : {SPATIAL_BLEND_WEIGHT_LGBM:.2f}")
    print(f"multipliers         : {MULTIPLIERS.tolist()}")
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
