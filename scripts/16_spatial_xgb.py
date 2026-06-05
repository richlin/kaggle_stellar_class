"""Phase 8 Task 23: spatial-aware XGBoost, then blend with the spatial LightGBM.

Reuses the cached spatial features from scripts/15_spatial_features.py, trains an
XGBoost (different family -> decorrelated errors) on baseline+spatial, and searches
a blend of the two spatial-aware models.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
SEEDS = [42, 43]
N_SPLITS = 5

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 2000,
    "learning_rate": 0.04,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "n_jobs": -1,
}

SP_TRAIN = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
SP_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
SP_LGBM_OOF = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
SP_LGBM_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
OOF_PROB = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
TEST_PROB = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "16_spatial_xgb.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "16_spatial_blend.csv"


def encode(X: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot the category columns, aligned across train/test, as float."""
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    Xe = pd.get_dummies(X, columns=cat, dtype=float)
    Xte = pd.get_dummies(X_test, columns=cat, dtype=float)
    Xte = Xte.reindex(columns=Xe.columns, fill_value=0.0)
    return Xe.to_numpy(np.float32), Xte.to_numpy(np.float32)


def run_xgb(Xe, y, Xte):
    oof = np.zeros((len(Xe), 3))
    test = np.zeros((len(Xte), 3))
    for seed in SEEDS:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(Xe, y), 1):
            print(f"  xgb seed {seed} fold {fold}/{N_SPLITS}")
            model = XGBClassifier(**XGB_PARAMS, random_state=seed)
            model.fit(
                Xe[tr], y[tr],
                sample_weight=compute_sample_weight("balanced", y[tr]),
                eval_set=[(Xe[va], y[va])],
                verbose=False,
            )
            oof[va] += model.predict_proba(Xe[va]) / len(SEEDS)
            test += model.predict_proba(Xte) / (len(SEEDS) * N_SPLITS)
    return oof, test


def main() -> int:
    train, test, sample = load_raw()
    X, y, _cat, encoder = build_features(train)
    X_test, _y, _c, _e = build_features(test, label_encoder=encoder)
    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    for j in range(sp_tr.shape[1]):
        X[f"sp{j}"] = sp_tr[:, j]
        X_test[f"sp{j}"] = sp_te[:, j]

    Xe, Xte = encode(X, X_test)
    print(f"xgb feature matrix: {Xe.shape}")
    oof, test_prob = run_xgb(Xe, y, Xte)
    np.save(OOF_PROB, oof)
    np.save(TEST_PROB, test_prob)

    _m, xgb_score = search_class_multipliers(y, oof)
    lgbm_oof = np.load(SP_LGBM_OOF)
    lgbm_test = np.load(SP_LGBM_TEST)

    # blend-weight search over the two spatial-aware models
    best = None
    for w in np.linspace(0, 1, 21):
        blend = w * lgbm_oof + (1 - w) * oof
        mult, score = search_class_multipliers(y, blend)
        if best is None or score > best["score"]:
            best = {"w_lgbm": float(w), "score": score, "mult": mult}
    blend_test = best["w_lgbm"] * lgbm_test + (1 - best["w_lgbm"]) * test_prob
    mult = best["mult"]
    pred = (best["w_lgbm"] * lgbm_oof + (1 - best["w_lgbm"]) * oof) * mult
    recalls = per_class_recall(y, pred.argmax(1), CLASS_LABELS)

    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(),
         "class": encoder.inverse_transform((blend_test * mult).argmax(1))}
    )
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)

    write_json(EXPERIMENT, {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "spatial_xgb_tuned_oof": xgb_score,
        "spatial_lgbm_tuned_oof": 0.968884,
        "blend_w_lgbm": best["w_lgbm"],
        "blend_tuned_oof": best["score"],
        "blend_multipliers": mult.tolist(),
        "blend_per_class_recall": recalls,
        "submission_path": str(SUBMISSION),
    })

    print("\n================ SPATIAL XGB + BLEND ================")
    print(f"spatial XGB tuned OOF  : {xgb_score:.6f}")
    print("spatial LGBM tuned OOF : 0.968884")
    print(f"best blend (w_lgbm={best['w_lgbm']:.2f}) : {best['score']:.6f}")
    print(f"  vs prior best 0.966282 : {best['score'] - 0.966282:+.6f}")
    print(f"blend per-class recall : {recalls}")
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
