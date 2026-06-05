"""Large-k spatial features: extend k∈{5,10,25,50,100,250} with k∈{1000,5000}.

The existing spatial features capture local sky structure (k≤250 neighbours).
Galaxy clusters and large-scale survey structure may be captured at wider scales.
This script adds k=1000 and k=5000 class-fraction features and trains a fresh
LightGBM.

The new k features are built OOF-safe using the same fold splits as script 15.

Acceptance gate: blend tuned OOF > 0.969202 (best honest OOF, script 41).
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
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
KS_LARGE = [1000, 5000]
MAX_K_LARGE = 5000
SMOOTHING = 10.0
SPATIAL_FOLD_SEED = 2024
SPATIAL_N_FOLDS = 5
CV_SEEDS = [42, 43, 44]
CV_N_SPLITS = 5
INCUMBENT_OOF = 0.969202

LGBM_PARAMS = {
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

SP_TRAIN = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
SP_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
SP_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
LARGE_K_TRAIN = PROJECT_ROOT / "experiments" / "42_large_k_train_features.npy"
LARGE_K_TEST = PROJECT_ROOT / "experiments" / "42_large_k_test_features.npy"
LARGE_K_NAMES = PROJECT_ROOT / "experiments" / "42_large_k_train_features.names.npy"
OOF_PROB = PROJECT_ROOT / "experiments" / "42_large_k_oof_probabilities.npy"
TEST_PROB = PROJECT_ROOT / "experiments" / "42_large_k_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "42_large_k_spatial.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "42_large_k_spatial.csv"


def build_large_k_features(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    if LARGE_K_TRAIN.exists() and LARGE_K_TEST.exists() and LARGE_K_NAMES.exists():
        print("reusing cached large-k spatial features")
        return (
            np.load(LARGE_K_TRAIN),
            np.load(LARGE_K_TEST),
            list(np.load(LARGE_K_NAMES, allow_pickle=True)),
        )

    xyz_tr = radec_to_xyz(train["alpha"].to_numpy(), train["delta"].to_numpy())
    xyz_te = radec_to_xyz(test["alpha"].to_numpy(), test["delta"].to_numpy())
    priors = np.bincount(y, minlength=3) / len(y)

    fold_ids = np.full(len(y), -1, dtype=np.int16)
    skf = StratifiedKFold(SPATIAL_N_FOLDS, shuffle=True, random_state=SPATIAL_FOLD_SEED)
    for f, (_, va) in enumerate(skf.split(xyz_tr, y)):
        fold_ids[va] = f

    print(f"building large-k spatial OOF features (k∈{KS_LARGE}) ...")
    train_feats, names = oof_neighbour_features(
        xyz_tr, y, fold_ids, KS_LARGE, 3, priors, SMOOTHING, MAX_K_LARGE
    )
    print("building large-k spatial test features ...")
    test_feats, _ = neighbour_features(xyz_te, xyz_tr, y, KS_LARGE, 3, priors, SMOOTHING, MAX_K_LARGE)

    np.save(LARGE_K_TRAIN, train_feats.astype(np.float32))
    np.save(LARGE_K_TEST, test_feats.astype(np.float32))
    np.save(LARGE_K_NAMES, np.array(names, dtype=object))
    return train_feats.astype(np.float32), test_feats.astype(np.float32), names


def run_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_cols: list[str]):
    oof = np.zeros((len(X), 3))
    test = np.zeros((len(X_test), 3))
    n_runs = len(CV_SEEDS)
    for seed in CV_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            print(f"  seed {seed} fold {fold}/{CV_N_SPLITS}")
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
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
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    # Load existing spatial features (k∈{5,10,25,50,100,250})
    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X[nm] = sp_tr[:, j]
        X_test[nm] = sp_te[:, j]

    # Build large-k features (k∈{1000,5000})
    lk_tr, lk_te, lk_names = build_large_k_features(train, test, y)
    for j, nm in enumerate(lk_names):
        X[nm] = lk_tr[:, j]
        X_test[nm] = lk_te[:, j]

    print(f"feature matrix: {X.shape} ({len(sp_names)} existing-spatial + {len(lk_names)} large-k features)")

    oof, test_prob = run_cv(X, y, X_test, cat_cols)

    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)

    np.save(OOF_PROB, oof)
    np.save(TEST_PROB, test_prob)

    # Also try blend with XGBoost
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)
    best_blend = None
    for w in np.linspace(0, 1, 21):
        blend = w * oof + (1 - w) * xgb_oof
        mult_b, score_b = search_class_multipliers(y, blend)
        if best_blend is None or score_b > best_blend["score"]:
            best_blend = {"w_lgbm": float(w), "score": score_b, "mult": mult_b}

    blend_test = best_blend["w_lgbm"] * test_prob + (1 - best_blend["w_lgbm"]) * xgb_test

    print("\n================ LARGE-K SPATIAL RESULT ================")
    print(f"standalone tuned OOF     : {tuned_score:.6f}")
    print(f"blend w_lgbm={best_blend['w_lgbm']:.2f} OOF : {best_blend['score']:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f} : {best_blend['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall         : {recalls}")

    gate = "PASSED" if best_blend["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "large_k_values": KS_LARGE,
        "max_k_large": MAX_K_LARGE,
        "standalone_tuned_oof": tuned_score,
        "blend_w_lgbm": best_blend["w_lgbm"],
        "blend_tuned_oof": best_blend["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": best_blend["mult"].tolist(),
        "per_class_recall": recalls,
        "params": LGBM_PARAMS,
        "cv_seeds": CV_SEEDS,
        "n_large_k_features": len(lk_names),
        "n_existing_spatial_features": len(sp_names),
    }

    if gate == "FAILED":
        print("\nFAILED acceptance gate — not writing submission")
        write_json(EXPERIMENT, record)
        return 0

    blend_mult = best_blend["mult"]
    predicted = (blend_test * blend_mult).argmax(1)
    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(predicted)}
    )
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)
    record["submission_path"] = str(SUBMISSION)
    write_json(EXPERIMENT, record)
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
