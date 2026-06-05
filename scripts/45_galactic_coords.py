"""Galactic coordinate features: add galactic l, b (and sin/cos) to the model.

Sky position in equatorial coordinates (RA, Dec) gives axis-aligned LightGBM
a complex curved decision boundary for the galactic plane. In galactic coords,
the galactic plane is simply |b| < threshold — a clean individual split.

Analysis shows: |b| < 10 deg → 98.9% STAR; |b| > 30 deg → 66.7% GALAXY.
This is a much stronger individual-feature signal than RA or Dec alone.

Since galactic and equatorial distances are identical (rotation on unit sphere),
galactic k-NN features add nothing — but galactic l,b as INDIVIDUAL features
give the model a direct axis-aligned cut on galactic structure.

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
from src.galactic import GALACTIC_FEATURE_NAMES, add_galactic_features
from src.validate import validate_submission
from src.validation import (
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
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
GAL_OOF_OUT = PROJECT_ROOT / "experiments" / "45_galactic_oof_probabilities.npy"
GAL_TEST_OUT = PROJECT_ROOT / "experiments" / "45_galactic_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "45_galactic_coords.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "45_galactic_coords.csv"


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

    # Add spatial features
    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X[nm] = sp_tr[:, j]
        X_test[nm] = sp_te[:, j]

    # Add galactic coordinate features
    print("Adding galactic coordinate features ...")
    X = add_galactic_features(X)
    X_test = add_galactic_features(X_test)
    gal_cols = GALACTIC_FEATURE_NAMES
    print(f"  added {len(gal_cols)} galactic features")

    print(f"feature matrix: {X.shape}")

    oof, test_prob = run_cv(X, y, X_test, cat_cols)

    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)

    np.save(GAL_OOF_OUT, oof)
    np.save(GAL_TEST_OUT, test_prob)

    # blend with XGBoost
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)
    best_blend = None
    for w in np.linspace(0, 1, 21):
        blend = w * oof + (1 - w) * xgb_oof
        mult_b, score_b = search_class_multipliers(y, blend)
        if best_blend is None or score_b > best_blend["score"]:
            best_blend = {"w_lgbm": float(w), "score": score_b, "mult": mult_b}

    blend_test = best_blend["w_lgbm"] * test_prob + (1 - best_blend["w_lgbm"]) * xgb_test

    print("\n================ GALACTIC COORD FEATURES RESULT ================")
    print(f"standalone tuned OOF     : {tuned_score:.6f}")
    print(f"blend w_lgbm={best_blend['w_lgbm']:.2f} OOF : {best_blend['score']:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f} : {best_blend['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall         : {recalls}")

    gate = "PASSED" if best_blend["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "galactic_features": gal_cols,
        "standalone_tuned_oof": tuned_score,
        "blend_w_lgbm": best_blend["w_lgbm"],
        "blend_tuned_oof": best_blend["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": best_blend["mult"].tolist(),
        "per_class_recall": recalls,
        "params": LGBM_PARAMS,
        "cv_seeds": CV_SEEDS,
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
