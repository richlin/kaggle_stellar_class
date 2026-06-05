"""Spatial pseudo-labeling: add high-confidence test predictions as training rows.

The current best blend (script 47) has 99%+ confidence on 75% of the test set.
These near-certain predictions are used as pseudo-labels. The existing spatial
features (15_spatial_test_features.npy) are kept fixed — no feature recomputation.

OOF is always evaluated on competition train rows ONLY. Pseudo-labeled test rows
are included only in the training portion of each CV fold.

Acceptance gate: tuned OOF > 0.969226 (best honest OOF, script 47).
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
INCUMBENT_OOF = 0.969226

# Confidence threshold: only include test rows with max blend prob above this.
CONFIDENCE_THRESH = 0.99
# Weight of pseudo-labeled rows relative to competition train rows (weight=1).
PSEUDO_WEIGHT = 0.5

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
# Blend test probs: LGBM5=0.4, XGB=0.4, CatBoost=0.2
LGBM5_TEST = PROJECT_ROOT / "experiments" / "32_spatial_5seed_lgbm_test_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
CAT_TEST = PROJECT_ROOT / "experiments" / "40_catboost_spatial_test_probabilities.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST_FILE = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
LGBM5_OOF = PROJECT_ROOT / "experiments" / "32_spatial_5seed_lgbm_oof_probabilities.npy"
CAT_OOF = PROJECT_ROOT / "experiments" / "40_catboost_spatial_oof_probabilities.npy"
OOF_PROB_OUT = PROJECT_ROOT / "experiments" / "50_pseudo_oof_probabilities.npy"
TEST_PROB_OUT = PROJECT_ROOT / "experiments" / "50_pseudo_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "50_spatial_pseudolabel.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "50_spatial_pseudolabel.csv"


def get_pseudo_labels(thresh: float, encoder) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Select high-confidence test rows and return (X_pseudo, y_pseudo, sp_pseudo)."""
    lgbm5_t = np.load(LGBM5_TEST)
    xgb_t = np.load(XGB_TEST)
    cat_t = np.load(CAT_TEST)
    blend_test = 0.4 * lgbm5_t + 0.4 * xgb_t + 0.2 * cat_t

    max_prob = blend_test.max(axis=1)
    mask = max_prob > thresh
    pseudo_y = encoder.transform(
        [CLASS_LABELS[c] for c in blend_test[mask].argmax(axis=1)]
    )
    sp_pseudo = np.load(SP_TEST)[mask]
    return mask, pseudo_y, sp_pseudo, blend_test[mask]


def run_cv_with_pseudo(
    X_comp: pd.DataFrame,
    y_comp: np.ndarray,
    X_pseudo: pd.DataFrame,
    y_pseudo: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    pseudo_weight: float,
):
    """Train on competition-fold + pseudo rows; evaluate OOF on competition rows only."""
    oof = np.zeros((len(X_comp), 3))
    test = np.zeros((len(X_test), 3))
    n_runs = len(CV_SEEDS)
    n_pseudo = len(X_pseudo)

    for seed in CV_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_comp, y_comp), 1):
            print(f"  seed {seed} fold {fold}/{CV_N_SPLITS}  (comp_train={len(tr)}, pseudo={n_pseudo})")
            X_train = pd.concat([X_comp.iloc[tr], X_pseudo], ignore_index=True)
            y_train = np.concatenate([y_comp[tr], y_pseudo])
            sw = np.concatenate([
                np.ones(len(tr), dtype=float),
                np.full(n_pseudo, pseudo_weight, dtype=float),
            ])
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(
                X_train, y_train,
                sample_weight=sw,
                eval_set=[(X_comp.iloc[va], y_comp[va])],
                eval_metric="multi_logloss",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            oof[va] += model.predict_proba(X_comp.iloc[va]) / n_runs
            test += model.predict_proba(X_test) / (n_runs * CV_N_SPLITS)
    return oof, test


def main() -> int:
    train, test, sample = load_raw()
    X_comp, y_comp, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X_comp[nm] = sp_tr[:, j]
        X_test[nm] = sp_te[:, j]

    # Build pseudo-labeled test rows using existing test spatial features
    print(f"Selecting pseudo-labels at confidence > {CONFIDENCE_THRESH} ...")
    mask, y_pseudo, sp_pseudo, pseudo_blend_probs = get_pseudo_labels(
        CONFIDENCE_THRESH, encoder
    )
    n_pseudo = mask.sum()
    print(f"  selected {n_pseudo:,} / {len(mask):,} test rows ({n_pseudo/len(mask)*100:.1f}%)")

    # Add spatial features to pseudo test rows using saved test spatial features
    test_with_spatial = X_test.copy()  # already has sp{j} columns from above
    X_pseudo = test_with_spatial[mask]

    print(f"Competition train: {len(X_comp)}, pseudo rows: {n_pseudo}")
    print(f"Feature matrix: {X_comp.shape}")

    oof, test_prob = run_cv_with_pseudo(
        X_comp, y_comp, X_pseudo, y_pseudo, X_test, cat_cols, PSEUDO_WEIGHT
    )

    mult, tuned_score = search_class_multipliers(y_comp, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y_comp, pred, CLASS_LABELS)

    np.save(OOF_PROB_OUT, oof)
    np.save(TEST_PROB_OUT, test_prob)

    # Blend with XGBoost for final candidate
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST_FILE)
    best_blend = None
    for w in np.linspace(0, 1, 21):
        blend = w * oof + (1 - w) * xgb_oof
        mult_b, score_b = search_class_multipliers(y_comp, blend)
        if best_blend is None or score_b > best_blend["score"]:
            best_blend = {"w_lgbm": float(w), "score": score_b, "mult": mult_b}

    blend_test = best_blend["w_lgbm"] * test_prob + (1 - best_blend["w_lgbm"]) * xgb_test

    print("\n================ SPATIAL PSEUDO-LABEL RESULT ================")
    print(f"standalone tuned OOF       : {tuned_score:.6f}")
    print(f"blend w_lgbm={best_blend['w_lgbm']:.2f} OOF  : {best_blend['score']:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f} : {best_blend['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall (standalone): {recalls}")
    print(f"n_pseudo ({CONFIDENCE_THRESH}) = {n_pseudo}")

    gate = "PASSED" if best_blend["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "confidence_thresh": CONFIDENCE_THRESH,
        "pseudo_weight": PSEUDO_WEIGHT,
        "n_pseudo_rows": int(n_pseudo),
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
