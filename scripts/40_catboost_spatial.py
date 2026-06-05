"""CatBoost with spatial features — genuinely different model family.

CatBoost uses symmetric (oblivious) trees and ordered boosting, giving a
materially different model structure from LightGBM and XGBoost. In Phase 5
(without spatial features), CatBoost got 0 blend weight. With spatial features,
it may add more useful diversity.

Acceptance gate: standalone tuned OOF > 0.967; blend OOF > 0.969154.
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
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
SEEDS = [42, 43]  # keep to 2 seeds given CatBoost training time
CV_N_SPLITS = 5
INCUMBENT_OOF = 0.969154  # best honest OOF from script 32

CAT_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "random_seed": 42,
    "task_type": "CPU",
    "verbose": 0,
    "early_stopping_rounds": 50,
    "thread_count": -1,
}

SP_TRAIN = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
SP_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
SP_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"
LGBM_OOF = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
LGBM_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
CAT_OOF_OUT = PROJECT_ROOT / "experiments" / "40_catboost_spatial_oof_probabilities.npy"
CAT_TEST_OUT = PROJECT_ROOT / "experiments" / "40_catboost_spatial_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "40_catboost_spatial.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "40_catboost_spatial_blend.csv"


def encode_for_catboost(X: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode categoricals for CatBoost (it needs feature names, not ints)."""
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    Xe = pd.get_dummies(X, columns=cat, dtype=float)
    Xte = pd.get_dummies(X_test, columns=cat, dtype=float)
    Xte = Xte.reindex(columns=Xe.columns, fill_value=0.0)
    return Xe, Xte


def run_catboost(Xe: pd.DataFrame, y: np.ndarray, Xte: pd.DataFrame):
    oof = np.zeros((len(Xe), 3))
    test = np.zeros((len(Xte), 3))
    n = len(SEEDS)
    for seed in SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(Xe, y), 1):
            print(f"  catboost seed {seed} fold {fold}/{CV_N_SPLITS}")
            sw = compute_sample_weight("balanced", y[tr])
            model = CatBoostClassifier(**{**CAT_PARAMS, "random_seed": seed})
            model.fit(
                Xe.iloc[tr], y[tr],
                sample_weight=sw,
                eval_set=(Xe.iloc[va], y[va]),
            )
            oof[va] += model.predict_proba(Xe.iloc[va]) / n
            test += model.predict_proba(Xte) / (n * CV_N_SPLITS)
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

    Xe, Xte = encode_for_catboost(X, X_test)
    print(f"catboost feature matrix: {Xe.shape}")

    if CAT_OOF_OUT.exists() and CAT_TEST_OUT.exists():
        print("reusing cached CatBoost spatial probs")
        cat_oof = np.load(CAT_OOF_OUT)
        cat_test = np.load(CAT_TEST_OUT)
    else:
        cat_oof, cat_test = run_catboost(Xe, y, Xte)
        np.save(CAT_OOF_OUT, cat_oof)
        np.save(CAT_TEST_OUT, cat_test)

    _m, cat_score = search_class_multipliers(y, cat_oof)

    lgbm_oof = np.load(LGBM_OOF)
    lgbm_test = np.load(LGBM_TEST)
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)

    # 3-model blend: LGBM + XGBoost + CatBoost
    best = None
    for wl in np.linspace(0, 1, 11):
        for wx in np.linspace(0, 1 - wl, 11):
            wc = 1 - wl - wx
            blend = wl * lgbm_oof + wx * xgb_oof + wc * cat_oof
            mult, score = search_class_multipliers(y, blend)
            if best is None or score > best["score"]:
                best = {"w_lgbm": float(wl), "w_xgb": float(wx), "w_cat": float(wc),
                        "score": score, "mult": mult}

    blend_oof = best["w_lgbm"] * lgbm_oof + best["w_xgb"] * xgb_oof + best["w_cat"] * cat_oof
    blend_test = best["w_lgbm"] * lgbm_test + best["w_xgb"] * xgb_test + best["w_cat"] * cat_test
    mult = best["mult"]
    recalls = per_class_recall(y, (blend_oof * mult).argmax(1), CLASS_LABELS)

    print("\n================ CATBOOST SPATIAL BLEND ================")
    print(f"CatBoost standalone tuned OOF : {cat_score:.6f}")
    print(f"3-model blend OOF             : {best['score']:.6f}")
    print(f"  w_lgbm={best['w_lgbm']:.2f} w_xgb={best['w_xgb']:.2f} w_cat={best['w_cat']:.2f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f}  : {best['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall              : {recalls}")

    gate = "PASSED" if best["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "catboost_standalone_oof": cat_score,
        "catboost_params": CAT_PARAMS,
        "catboost_seeds": SEEDS,
        "blend_w_lgbm": best["w_lgbm"],
        "blend_w_xgb": best["w_xgb"],
        "blend_w_cat": best["w_cat"],
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
