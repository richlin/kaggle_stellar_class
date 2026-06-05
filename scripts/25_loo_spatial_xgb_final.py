"""Final-only LOO spatial XGBoost candidate.

`19_loo_spatial_final.csv` improved public score by training the LightGBM side
on leave-one-out spatial train features. The XGBoost side in that blend still
uses the old KFold-OOF spatial train feature distribution. This script trains a
full-data XGBoost model on the same LOO spatial features and blends it with the
LOO LightGBM probabilities.
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
from xgboost import XGBClassifier

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import write_json

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
SEEDS = [42, 43]
BLEND_WEIGHT_LGBM = 0.55
MULTIPLIERS = np.array([0.45, 0.75, 1.0])

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 900,
    "learning_rate": 0.04,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "tree_method": "hist",
    "n_jobs": -1,
}

LOO_TRAIN = PROJECT_ROOT / "experiments" / "19_loo_spatial_train_features.npy"
TEST_SPATIAL = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
LOO_FEATURE_NAMES = PROJECT_ROOT / "experiments" / "19_loo_spatial_feature_names.npy"
LOO_LGBM_TEST = PROJECT_ROOT / "experiments" / "19_loo_spatial_lgbm_test_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "25_loo_spatial_xgb_test_probabilities.npy"
BLEND_TEST = PROJECT_ROOT / "experiments" / "25_loo_spatial_xgb_blend_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "25_loo_spatial_xgb_final.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "25_loo_spatial_xgb_final.csv"


def encode(X: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode categoricals consistently for XGBoost."""
    cat_cols = [col for col in X.columns if str(X[col].dtype) == "category"]
    train_encoded = pd.get_dummies(X, columns=cat_cols, dtype=float)
    test_encoded = pd.get_dummies(X_test, columns=cat_cols, dtype=float)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
    return train_encoded.to_numpy(np.float32), test_encoded.to_numpy(np.float32)


def add_loo_features(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append cached LOO train spatial features and full-train test features."""
    names = list(np.load(LOO_FEATURE_NAMES, allow_pickle=True))
    train_features = np.load(LOO_TRAIN)
    test_features = np.load(TEST_SPATIAL)
    train_frame = pd.DataFrame(train_features, columns=names, index=X.index)
    test_frame = pd.DataFrame(test_features, columns=names, index=X_test.index)
    return pd.concat([X, train_frame], axis=1), pd.concat([X_test, test_frame], axis=1)


def train_full_xgb(X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Average full-data XGBoost test probabilities across seeds."""
    test_prob = np.zeros((len(X_test), len(CLASS_LABELS)), dtype=float)
    for seed in SEEDS:
        print(f"  full LOO XGBoost seed {seed}")
        model = XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X, y, verbose=False)
        test_prob += model.predict_proba(X_test) / len(SEEDS)
    return test_prob


def make_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder,
) -> pd.DataFrame:
    """Build a competition submission while preserving sample id order."""
    predicted = (probabilities * multipliers).argmax(axis=1)
    return pd.DataFrame(
        {
            "id": sample_submission["id"].to_numpy(),
            "class": encoder.inverse_transform(predicted),
        }
    )


def main() -> int:
    train, test, sample = load_raw()
    X, y, _cat, encoder = build_features(train)
    if y is None:
        raise ValueError("training data must include class labels")
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)
    X, X_test = add_loo_features(X, X_test)
    train_array, test_array = encode(X, X_test)
    xgb_test = train_full_xgb(train_array, y, test_array)
    np.save(XGB_TEST, xgb_test)

    blend = BLEND_WEIGHT_LGBM * np.load(LOO_LGBM_TEST) + (1 - BLEND_WEIGHT_LGBM) * xgb_test
    np.save(BLEND_TEST, blend)
    submission = make_submission(sample, blend, MULTIPLIERS, encoder)
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
                "Train the XGBoost side on leave-one-out spatial train features after "
                "public feedback confirmed the LOO LightGBM final-feature-density gain."
            ),
            "seeds": SEEDS,
            "params": XGB_PARAMS,
            "blend_weight_lgbm": BLEND_WEIGHT_LGBM,
            "multipliers": MULTIPLIERS.tolist(),
            "public_score": None,
            "submission_path": str(SUBMISSION),
        },
    )
    print("\n================ LOO SPATIAL XGB FINAL CANDIDATE ================")
    print("honest OOF          : n/a (final-only train/test mismatch candidate)")
    print(f"blend weight LGBM   : {BLEND_WEIGHT_LGBM:.2f}")
    print(f"multipliers         : {MULTIPLIERS.tolist()}")
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
