"""Spatial LightGBM with more trees (n_estimators=1500, early stopping=50).

Both spatial-only and spatial+photometric single-fold models hit max trees at
n_estimators=900. This script runs spatial LGBM with increased capacity
(n_estimators=1500) to see if the model improves with more iterations.

Acceptance gate: blend tuned OOF > 0.969071 (best honest OOF, 16_spatial_blend).
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
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
CV_SEEDS = [42, 43, 44]
CV_N_SPLITS = 5
INCUMBENT_OOF = 0.969071

LGBM_PARAMS = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 1500,  # increased from 900
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
LGBM_OOF_OUT = PROJECT_ROOT / "experiments" / "36_spatial_1500trees_oof_probabilities.npy"
LGBM_TEST_OUT = PROJECT_ROOT / "experiments" / "36_spatial_1500trees_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "36_spatial_more_trees.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "36_spatial_more_trees.csv"


def run_cv(X, y, X_test, cat_cols):
    oof = np.zeros((len(X), 3))
    test = np.zeros((len(X_test), 3))
    best_iters = []
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
            best_iters.append(model.best_iteration_)
    print(f"  best_iterations: min={min(best_iters)} max={max(best_iters)} mean={sum(best_iters)/len(best_iters):.0f}")
    return oof, test


def main() -> int:
    train, test, sample = load_raw()
    X, y, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X[nm] = sp_tr[:, j]
        X_test[nm] = sp_te[:, j]

    # load from cache if available
    if LGBM_OOF_OUT.exists() and LGBM_TEST_OUT.exists():
        print("reusing cached 1500-tree spatial LGBM")
        lgbm_oof = np.load(LGBM_OOF_OUT)
        lgbm_test = np.load(LGBM_TEST_OUT)
    else:
        lgbm_oof, lgbm_test = run_cv(X, y, X_test, cat_cols)
        np.save(LGBM_OOF_OUT, lgbm_oof)
        np.save(LGBM_TEST_OUT, lgbm_test)

    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)

    _m, lgbm_score = search_class_multipliers(y, lgbm_oof)

    best = None
    for w in np.linspace(0, 1, 21):
        blend = w * lgbm_oof + (1 - w) * xgb_oof
        mult, score = search_class_multipliers(y, blend)
        if best is None or score > best["score"]:
            best = {"w_lgbm": float(w), "score": score, "mult": mult}

    blend_test = best["w_lgbm"] * lgbm_test + (1 - best["w_lgbm"]) * xgb_test
    mult = best["mult"]
    recalls = per_class_recall(y, (blend_test * mult).argmax(1), CLASS_LABELS)

    print("\n================ 1500-TREE SPATIAL BLEND ================")
    print(f"LGBM standalone tuned OOF : {lgbm_score:.6f}")
    print(f"blend w_lgbm={best['w_lgbm']:.2f} OOF   : {best['score']:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f} : {best['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall          : {recalls}")

    gate = "PASSED" if best["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "n_estimators": LGBM_PARAMS["n_estimators"],
        "lgbm_standalone_oof": lgbm_score,
        "blend_w_lgbm": best["w_lgbm"],
        "blend_tuned_oof": best["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": mult.tolist(),
        "per_class_recall": recalls,
        "params": LGBM_PARAMS,
        "cv_seeds": CV_SEEDS,
    }

    if gate == "FAILED":
        print("\nFAILED acceptance gate — not writing submission")
        write_json(EXPERIMENT, record)
        return 0

    predicted = (blend_test * mult).argmax(1)
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
