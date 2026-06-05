"""Three-base-model meta-stacker: spatial LGBM + spatial XGBoost + photometric LGBM.

Requires scripts 15, 16, and 28 to have been run first.

Acceptance gate: tuned OOF > 0.969071 (best honest OOF, 16_spatial_blend).
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
from src.validate import validate_submission
from src.validation import (
    balanced_accuracy,
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
INCUMBENT_OOF = 0.969071
META_CV_SEED = 0
META_N_SPLITS = 5

META_PARAMS = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
}

LGBM_OOF = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
LGBM_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
PHOT_OOF = PROJECT_ROOT / "experiments" / "28_phot_oof_probabilities.npy"
PHOT_TEST = PROJECT_ROOT / "experiments" / "28_phot_test_probabilities.npy"
SP_TRAIN = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
SP_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
SP_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"
META_OOF_OUT = PROJECT_ROOT / "experiments" / "34_3model_meta_oof_probabilities.npy"
META_TEST_OUT = PROJECT_ROOT / "experiments" / "34_3model_meta_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "34_3model_meta_stacker.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "34_3model_meta_stacker.csv"

EPS = 1e-8


def to_logits(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1))


def run_meta_cv(meta_tr: np.ndarray, y: np.ndarray, meta_te: np.ndarray):
    oof = np.zeros((len(meta_tr), 3))
    test = np.zeros((len(meta_te), 3))
    skf = StratifiedKFold(META_N_SPLITS, shuffle=True, random_state=META_CV_SEED)
    for fold, (tr, va) in enumerate(skf.split(meta_tr, y), 1):
        print(f"  meta fold {fold}/{META_N_SPLITS}")
        model = LGBMClassifier(**META_PARAMS, random_state=META_CV_SEED)
        model.fit(
            meta_tr[tr], y[tr],
            eval_set=[(meta_tr[va], y[va])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        oof[va] = model.predict_proba(meta_tr[va])
        test += model.predict_proba(meta_te) / META_N_SPLITS
    return oof, test


def main() -> int:
    for req in (LGBM_OOF, XGB_OOF, PHOT_OOF):
        if not req.exists():
            raise FileNotFoundError(f"Required file {req.name} not found. Run scripts 15, 16, 28 first.")

    train, test, sample = load_raw()
    X, y, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    # spatial features for raw baseline
    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X[nm] = sp_tr[:, j]
        X_test[nm] = sp_te[:, j]

    # build meta features: logits from 3 base models + raw numeric
    lgbm_oof = np.load(LGBM_OOF)
    lgbm_test = np.load(LGBM_TEST)
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)
    phot_oof = np.load(PHOT_OOF)
    phot_test = np.load(PHOT_TEST)

    numeric_cols = [c for c in X.columns if c not in cat_cols]
    raw_tr = X[numeric_cols].to_numpy(np.float32)
    raw_te = X_test[numeric_cols].to_numpy(np.float32)

    meta_tr = np.hstack([
        to_logits(lgbm_oof), to_logits(xgb_oof), to_logits(phot_oof), raw_tr
    ]).astype(np.float32)
    meta_te = np.hstack([
        to_logits(lgbm_test), to_logits(xgb_test), to_logits(phot_test), raw_te
    ]).astype(np.float32)

    print(f"meta feature matrix: {meta_tr.shape} (9 logit + {raw_tr.shape[1]} numeric)")
    oof, test_prob = run_meta_cv(meta_tr, y, meta_te)

    argmax_score = balanced_accuracy(y, oof.argmax(1))
    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)

    np.save(META_OOF_OUT, oof)
    np.save(META_TEST_OUT, test_prob)

    print("\n================ 3-MODEL META-STACKER ================")
    print(f"argmax OOF                       : {argmax_score:.6f}")
    print(f"tuned  OOF                       : {tuned_score:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f}    : {tuned_score - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall (tuned)         : {recalls}")
    print(f"chosen multipliers               : {mult.round(4).tolist()}")

    gate = "PASSED" if tuned_score > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "base_models": ["spatial_lgbm_15", "spatial_xgb_16", "photometric_lgbm_28"],
        "meta_features": "9_logits + raw_numeric",
        "meta_cv_seed": META_CV_SEED,
        "meta_n_splits": META_N_SPLITS,
        "meta_params": META_PARAMS,
        "argmax_oof": argmax_score,
        "tuned_oof": tuned_score,
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": mult.tolist(),
        "per_class_recall": recalls,
    }

    if gate == "FAILED":
        print(f"\nFAILED acceptance gate ({INCUMBENT_OOF:.6f}) — not writing submission")
        write_json(EXPERIMENT, record)
        return 0

    test_pred = (test_prob * mult).argmax(1)
    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(test_pred)}
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
