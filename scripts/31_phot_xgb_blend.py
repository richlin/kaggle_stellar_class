"""Task 31 (Phase 11 / revisit plan): XGBoost trained on spatial + photometric features.

Mirrors 16_spatial_xgb.py: loads cached photometric-neighbour features from
script 28, trains XGBoost, and blends with the photometric LGBM OOF probs.

Requires scripts/28_photometric_neighbours.py to have been run first.

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
INCUMBENT_OOF = 0.969071

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
SP_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"
PHOT_TRAIN = PROJECT_ROOT / "experiments" / "28_phot_train_features.npy"
PHOT_TEST = PROJECT_ROOT / "experiments" / "28_phot_test_features.npy"
PHOT_NAMES = PROJECT_ROOT / "experiments" / "28_phot_train_features.names.npy"
LGBM_OOF = PROJECT_ROOT / "experiments" / "28_phot_oof_probabilities.npy"
LGBM_TEST = PROJECT_ROOT / "experiments" / "28_phot_test_probabilities.npy"
XGB_OOF_OUT = PROJECT_ROOT / "experiments" / "31_phot_xgb_oof_probabilities.npy"
XGB_TEST_OUT = PROJECT_ROOT / "experiments" / "31_phot_xgb_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "31_phot_xgb_blend.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "31_phot_xgb_blend.csv"


def encode(X: pd.DataFrame, X_test: pd.DataFrame):
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
    for req in (SP_TRAIN, PHOT_TRAIN, LGBM_OOF):
        if not req.exists():
            raise FileNotFoundError(
                f"Required file {req} not found. Run scripts 15 and 28 first."
            )

    train, test, sample = load_raw()
    X, y, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    phot_tr = np.load(PHOT_TRAIN)
    phot_te = np.load(PHOT_TEST)
    phot_names = list(np.load(PHOT_NAMES, allow_pickle=True))

    extra_tr = np.hstack([sp_tr, phot_tr])
    extra_te = np.hstack([sp_te, phot_te])
    extra_names = sp_names + phot_names

    X = pd.concat([X, pd.DataFrame(extra_tr, columns=extra_names, index=X.index)], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(extra_te, columns=extra_names, index=X_test.index)], axis=1)

    Xe, Xte = encode(X, X_test)
    print(f"xgb feature matrix: {Xe.shape}")

    xgb_oof, xgb_test = run_xgb(Xe, y, Xte)
    np.save(XGB_OOF_OUT, xgb_oof)
    np.save(XGB_TEST_OUT, xgb_test)

    _m, xgb_score = search_class_multipliers(y, xgb_oof)
    lgbm_oof = np.load(LGBM_OOF)
    lgbm_test = np.load(LGBM_TEST)

    best = None
    for w in np.linspace(0, 1, 21):
        blend = w * lgbm_oof + (1 - w) * xgb_oof
        mult, score = search_class_multipliers(y, blend)
        if best is None or score > best["score"]:
            best = {"w_lgbm": float(w), "score": score, "mult": mult}

    blend_oof = best["w_lgbm"] * lgbm_oof + (1 - best["w_lgbm"]) * xgb_oof
    blend_test = best["w_lgbm"] * lgbm_test + (1 - best["w_lgbm"]) * xgb_test
    mult = best["mult"]
    recalls = per_class_recall(y, (blend_oof * mult).argmax(1), CLASS_LABELS)

    print("\n================ PHOT XGB + BLEND ================")
    print(f"phot XGB tuned OOF    : {xgb_score:.6f}")
    print(f"best blend w_lgbm={best['w_lgbm']:.2f} : {best['score']:.6f}")
    print(f"  vs incumbent 0.969071 : {best['score'] - INCUMBENT_OOF:+.6f}")
    print(f"blend per-class recall : {recalls}")

    gate = "PASSED" if best["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "xgb_tuned_oof": xgb_score,
        "blend_w_lgbm": best["w_lgbm"],
        "blend_tuned_oof": best["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "blend_multipliers": mult.tolist(),
        "blend_per_class_recall": recalls,
        "xgb_params": XGB_PARAMS,
        "seeds": SEEDS,
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
