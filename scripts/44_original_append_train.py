"""Task 48: train on competition train + audited original data; OOF on competition rows only.

The original data rows are added to the training fold only — they are never in
the validation fold, so the OOF is still a valid competition-domain estimate.

Requires:
  - scripts/43_original_append_audit.py to have been run with PASS verdict
  - The original dataset path used in the audit

Acceptance gate: competition-domain OOF > 0.969202 (best honest OOF, script 41).
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
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
SP_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
OOF_PROB_OUT = PROJECT_ROOT / "experiments" / "44_append_lgbm_oof_probabilities.npy"
TEST_PROB_OUT = PROJECT_ROOT / "experiments" / "44_append_lgbm_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "44_original_append_train.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "44_original_append.csv"


def _derive_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive spectral_type and galaxy_population from photometry."""
    df = df.copy()
    df["spectral_type"] = pd.cut(
        df["r"] - df["g"],
        bins=[-np.inf, -1, -0.5, 0, np.inf],
        labels=["M", "G/K", "A/F", "O/B"],
    )
    df["galaxy_population"] = pd.cut(
        df["u"] - df["r"],
        bins=[-np.inf, 2.2, np.inf],
        labels=["Blue_Cloud", "Red_Sequence"],
    )
    return df


def load_and_prepare_original(path: str, encoder) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the audited original dataset and build features compatible with competition."""
    df = pd.read_csv(path)
    # Normalise class column
    for col in ["class", "Class", "CLASS", "label"]:
        if col in df.columns:
            df = df.rename(columns={col: "class"})
            break
    cls_map = {}
    for v in df["class"].unique():
        vstr = str(v).upper().strip()
        if vstr in {"GALAXY", "QSO", "STAR"}:
            cls_map[v] = vstr
        elif "GAL" in vstr:
            cls_map[v] = "GALAXY"
        elif "QSO" in vstr or "QUASAR" in vstr:
            cls_map[v] = "QSO"
        elif "STAR" in vstr or "STELLAR" in vstr:
            cls_map[v] = "STAR"
    df["class"] = df["class"].map(cls_map).dropna()
    df = df.dropna(subset=["class"])

    # Derive spectral_type and galaxy_population if missing
    if "spectral_type" not in df.columns or "galaxy_population" not in df.columns:
        df = _derive_categoricals(df)

    X_orig, y_orig, _, _ = build_features(df, label_encoder=encoder)
    return X_orig, y_orig


def run_cv_with_append(
    X_comp: pd.DataFrame,
    y_comp: np.ndarray,
    X_orig: pd.DataFrame,
    y_orig: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
):
    """
    Train on (competition training fold) + (all original rows).
    OOF predictions are computed ONLY on competition validation rows.
    """
    oof = np.zeros((len(X_comp), 3))
    test = np.zeros((len(X_test), 3))
    n_runs = len(CV_SEEDS)
    n_orig = len(X_orig)

    for seed in CV_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_comp, y_comp), 1):
            print(f"  seed {seed} fold {fold}/{CV_N_SPLITS} (comp train={len(tr)}, orig={n_orig})")
            # Combine competition training fold + all original rows
            X_train = pd.concat([X_comp.iloc[tr], X_orig], ignore_index=True)
            y_train = np.concatenate([y_comp[tr], y_orig])

            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(
                X_train, y_train,
                eval_set=[(X_comp.iloc[va], y_comp[va])],
                eval_metric="multi_logloss",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            oof[va] += model.predict_proba(X_comp.iloc[va]) / n_runs
            test += model.predict_proba(X_test) / (n_runs * CV_N_SPLITS)

    return oof, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to audited original dataset CSV")
    args = parser.parse_args()

    # Verify audit passed
    audit_path = PROJECT_ROOT / "experiments" / "43_original_append_audit.json"
    if not audit_path.exists():
        print("ERROR: Run scripts/43_original_append_audit.py first")
        return 1
    import json
    audit = json.loads(audit_path.read_text())
    if audit.get("verdict") != "PASS":
        print(f"ERROR: audit verdict is {audit.get('verdict')} — cannot proceed")
        return 1
    print(f"Audit PASSED: {audit.get('clean_append_rows', '?')} original rows")

    # Load competition data
    train, test, sample = load_raw()
    X_comp, y_comp, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    # Add spatial features to competition data
    sp_tr = np.load(SP_TRAIN)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X_comp[nm] = sp_tr[:, j]

    # Load original data
    print(f"Loading original data from {args.original} ...")
    X_orig, y_orig = load_and_prepare_original(args.original, encoder)
    # Add spatial features for original data (using full competition train as reference)
    from src.spatial import neighbour_features, radec_to_xyz
    xyz_comp = radec_to_xyz(train["alpha"].to_numpy(), train["delta"].to_numpy())
    xyz_orig = radec_to_xyz(X_orig["alpha"].to_numpy(), X_orig["delta"].to_numpy())
    priors = np.bincount(y_comp, minlength=3) / len(y_comp)
    sp_orig, _ = neighbour_features(xyz_orig, xyz_comp, y_comp, [5, 10, 25, 50, 100, 250], 3, priors, 10.0, 250)
    for j, nm in enumerate(sp_names):
        X_orig[nm] = sp_orig[:, j]

    print(f"Competition train: {len(X_comp)} rows")
    print(f"Original data:     {len(X_orig)} rows")
    print(f"Combined training: {len(X_comp) + len(X_orig)} rows")
    print(f"Feature matrix:    {X_comp.shape[1]} features")

    # Add spatial features for test
    sp_te = np.load(PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy")
    for j, nm in enumerate(sp_names):
        X_test[nm] = sp_te[:, j]

    oof, test_prob = run_cv_with_append(X_comp, y_comp, X_orig, y_orig, X_test, cat_cols)

    mult, tuned_score = search_class_multipliers(y_comp, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y_comp, pred, CLASS_LABELS)

    np.save(OOF_PROB_OUT, oof)
    np.save(TEST_PROB_OUT, test_prob)

    # blend with XGBoost
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)
    best_blend = None
    for w in np.linspace(0, 1, 21):
        blend = w * oof + (1 - w) * xgb_oof
        mult_b, score_b = search_class_multipliers(y_comp, blend)
        if best_blend is None or score_b > best_blend["score"]:
            best_blend = {"w_lgbm": float(w), "score": score_b, "mult": mult_b}

    blend_test = best_blend["w_lgbm"] * test_prob + (1 - best_blend["w_lgbm"]) * xgb_test

    print("\n================ APPEND-DATA TRAINING RESULT ================")
    print(f"standalone tuned OOF         : {tuned_score:.6f}")
    print(f"blend w_lgbm={best_blend['w_lgbm']:.2f} OOF  : {best_blend['score']:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f} : {best_blend['score'] - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall (standalone): {recalls}")

    gate = "PASSED" if best_blend["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "original_path": args.original,
        "original_rows": len(X_orig),
        "competition_train_rows": len(X_comp),
        "combined_rows": len(X_comp) + len(X_orig),
        "standalone_tuned_oof": tuned_score,
        "blend_w_lgbm": best_blend["w_lgbm"],
        "blend_tuned_oof": best_blend["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": best_blend["mult"].tolist(),
        "per_class_recall": recalls,
        "params": LGBM_PARAMS,
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
