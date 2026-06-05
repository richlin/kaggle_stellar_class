"""Phase 8: spatial neighbourhood features -> CV LightGBM -> tuned submission.

Builds leakage-safe out-of-fold spatial k-NN class-fraction features on sky
position, concatenates them onto the baseline feature frame, runs a repeated-seed
5-fold LightGBM, tunes per-class multipliers on OOF, and writes a submission.

Spatial features are cached to .npy so re-runs only redo the model.
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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold

from src.data import build_features, load_raw
from src.spatial import neighbour_features, oof_neighbour_features, radec_to_xyz
from src.validate import validate_submission
from src.validation import (
    balanced_accuracy,
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
KS = [5, 10, 25, 50, 100, 250]
MAX_K = 250
SMOOTHING = 10.0
SPATIAL_FOLD_SEED = 2024
SPATIAL_N_FOLDS = 5
CV_SEEDS = [42, 43, 44]
CV_N_SPLITS = 5

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

OOF_SPATIAL = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
TEST_SPATIAL = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
OOF_PROB = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
TEST_PROB = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "15_spatial.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "15_spatial.csv"


def build_spatial(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    if OOF_SPATIAL.exists() and TEST_SPATIAL.exists():
        print("reusing cached spatial features")
        names = list(np.load(OOF_SPATIAL.with_suffix(".names.npy"), allow_pickle=True))
        return np.load(OOF_SPATIAL), np.load(TEST_SPATIAL), names

    xyz_tr = radec_to_xyz(train["alpha"].to_numpy(), train["delta"].to_numpy())
    xyz_te = radec_to_xyz(test["alpha"].to_numpy(), test["delta"].to_numpy())
    priors = np.bincount(y, minlength=3) / len(y)

    fold_ids = np.full(len(y), -1, dtype=np.int16)
    skf = StratifiedKFold(SPATIAL_N_FOLDS, shuffle=True, random_state=SPATIAL_FOLD_SEED)
    for f, (_t, va) in enumerate(skf.split(xyz_tr, y)):
        fold_ids[va] = f

    print("building OOF spatial features for train ...")
    train_feats, names = oof_neighbour_features(
        xyz_tr, y, fold_ids, KS, 3, priors, SMOOTHING, MAX_K
    )
    print("building spatial features for test ...")
    test_feats, _ = neighbour_features(xyz_te, xyz_tr, y, KS, 3, priors, SMOOTHING, MAX_K)

    np.save(OOF_SPATIAL, train_feats)
    np.save(TEST_SPATIAL, test_feats)
    np.save(OOF_SPATIAL.with_suffix(".names.npy"), np.array(names, dtype=object))
    return train_feats, test_feats, names


def run_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_cols: list[str]):
    oof = np.zeros((len(X), 3))
    test = np.zeros((len(X_test), 3))
    n_runs = len(CV_SEEDS)
    for seed in CV_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            print(f"  seed {seed} fold {fold}/{CV_N_SPLITS}")
            model = LGBMClassifier(**PARAMS, random_state=seed)
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[va], y[va])],
                eval_metric="multi_logloss",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            oof[va] += model.predict_proba(X.iloc[va]) / n_runs
            test += model.predict_proba(X_test) / (n_runs * CV_N_SPLITS)
    return oof, test


def main() -> int:
    train, test, sample = load_raw()
    X, y, cat_cols, encoder = build_features(train)
    X_test, _y, _c, _e = build_features(test, label_encoder=encoder)

    train_feats, test_feats, names = build_spatial(train, test, y)
    for j, nm in enumerate(names):
        X[nm] = train_feats[:, j]
        X_test[nm] = test_feats[:, j]

    print(f"feature matrix: {X.shape} ({len(names)} spatial features added)")
    oof, test_prob = run_cv(X, y, X_test, cat_cols)

    argmax_score = balanced_accuracy(y, oof.argmax(1))
    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)

    np.save(OOF_PROB, oof)
    np.save(TEST_PROB, test_prob)
    test_pred = (test_prob * mult).argmax(1)
    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(test_pred)}
    )
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)

    record = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "n_spatial_features": len(names),
        "spatial_feature_names": names,
        "ks": KS,
        "smoothing": SMOOTHING,
        "params": PARAMS,
        "cv_seeds": CV_SEEDS,
        "argmax_oof_balanced_accuracy": argmax_score,
        "tuned_oof_balanced_accuracy": tuned_score,
        "chosen_multipliers": mult.tolist(),
        "per_class_recall_tuned": recalls,
        "reference_best_oof": 0.966282,
        "submission_path": str(SUBMISSION),
    }
    write_json(EXPERIMENT, record)

    print("\n================ SPATIAL FEATURES RESULT ================")
    print(f"argmax OOF balanced accuracy : {argmax_score:.6f}")
    print(f"tuned  OOF balanced accuracy : {tuned_score:.6f}")
    print(f"  vs prior best 0.966282     : {tuned_score - 0.966282:+.6f}")
    print(f"per-class recall (tuned)     : {recalls}")
    print(f"chosen multipliers           : {mult.round(4).tolist()}")
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
