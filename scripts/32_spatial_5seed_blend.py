"""Task 31 variant: extend spatial LightGBM from 3 seeds to 5 seeds.

Trains seeds 45 and 46 on the same spatial features as script 15, averages
all five seed OOF/test probs, then re-blends with the 2-seed spatial XGBoost
from script 16.

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
EXISTING_SEEDS = [42, 43, 44]  # already trained in script 15
NEW_SEEDS = [45, 46]           # adding these
CV_N_SPLITS = 5
INCUMBENT_OOF = 0.969071

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
EXISTING_LGBM_OOF = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
EXISTING_LGBM_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
NEW_LGBM_OOF = PROJECT_ROOT / "experiments" / "32_spatial_5seed_lgbm_oof_probabilities.npy"
NEW_LGBM_TEST = PROJECT_ROOT / "experiments" / "32_spatial_5seed_lgbm_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "32_spatial_5seed_blend.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "32_spatial_5seed_blend.csv"


def run_new_seeds(X, y, X_test, cat_cols):
    """Train new seeds and return per-seed OOF/test accumulated as averages."""
    new_oof = np.zeros((len(X), 3))
    new_test = np.zeros((len(X_test), 3))
    n = len(NEW_SEEDS)
    for seed in NEW_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            print(f"  lgbm seed {seed} fold {fold}/{CV_N_SPLITS}")
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[va], y[va])],
                eval_metric="multi_logloss",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            new_oof[va] += model.predict_proba(X.iloc[va]) / n
            new_test += model.predict_proba(X_test) / (n * CV_N_SPLITS)
    return new_oof, new_test


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

    # train new seeds (or reload cached)
    if NEW_LGBM_OOF.exists() and NEW_LGBM_TEST.exists():
        print("reusing cached 5-seed extension")
        new_oof = np.load(NEW_LGBM_OOF)
        new_test = np.load(NEW_LGBM_TEST)
    else:
        new_oof, new_test = run_new_seeds(X, y, X_test, cat_cols)
        np.save(NEW_LGBM_OOF, new_oof)
        np.save(NEW_LGBM_TEST, new_test)

    # combine 3-seed existing average with 2 new seeds → 5-seed average
    existing_oof = np.load(EXISTING_LGBM_OOF)
    existing_test = np.load(EXISTING_LGBM_TEST)
    n_exist = len(EXISTING_SEEDS)
    n_new = len(NEW_SEEDS)
    lgbm_oof_5 = (n_exist * existing_oof + n_new * new_oof) / (n_exist + n_new)
    lgbm_test_5 = (n_exist * existing_test + n_new * new_test) / (n_exist + n_new)

    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)

    best = None
    for w in np.linspace(0, 1, 21):
        blend = w * lgbm_oof_5 + (1 - w) * xgb_oof
        mult, score = search_class_multipliers(y, blend)
        if best is None or score > best["score"]:
            best = {"w_lgbm": float(w), "score": score, "mult": mult}

    blend_oof = best["w_lgbm"] * lgbm_oof_5 + (1 - best["w_lgbm"]) * xgb_oof
    blend_test = best["w_lgbm"] * lgbm_test_5 + (1 - best["w_lgbm"]) * xgb_test
    mult = best["mult"]
    recalls = per_class_recall(y, (blend_oof * mult).argmax(1), CLASS_LABELS)

    print("\n================ 5-SEED SPATIAL BLEND ================")
    print(f"5-seed blend w_lgbm={best['w_lgbm']:.2f} OOF : {best['score']:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f}       : {best['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall                  : {recalls}")

    gate = "PASSED" if best["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "existing_seeds": EXISTING_SEEDS,
        "new_seeds": NEW_SEEDS,
        "blend_w_lgbm": best["w_lgbm"],
        "blend_tuned_oof": best["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": mult.tolist(),
        "per_class_recall": recalls,
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
