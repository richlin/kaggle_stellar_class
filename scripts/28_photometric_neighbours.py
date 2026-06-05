"""Task 2 (revisit plan): local photometric-neighbour features.

Extends the spatial-feature paradigm to three photometric spaces:
  space A — colour [u_g, g_r, r_i, i_z] StandardScaled (4D)
  space B — magnitude [u, g, r, i, z] StandardScaled (5D)
  space C — sphere (x,y,z) + colour StandardScaled (7D)

Builds leakage-safe OOF k-NN class-fraction features, combines with the
cached position-only spatial features from script 15, and trains a fresh
3-seed × 5-fold LightGBM with early stopping.

Acceptance gate: tuned OOF > 0.969071 (best honest OOF, 16_spatial_xgb).
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
from sklearn.preprocessing import StandardScaler

from src.data import build_features, load_raw
from src.spatial import neighbour_features, oof_neighbour_features, radec_to_xyz
from src.validate import validate_submission
from src.validation import (
    balanced_accuracy,
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
KS_PHOT = [5, 10, 25, 50]
MAX_K_PHOT = 50
SMOOTHING = 10.0
SPATIAL_FOLD_SEED = 2024
SPATIAL_N_FOLDS = 5
CV_SEEDS = [42, 43, 44]
CV_N_SPLITS = 5
INCUMBENT_OOF = 0.969071  # best honest OOF from 16_spatial_blend

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
PHOT_TRAIN = PROJECT_ROOT / "experiments" / "28_phot_train_features.npy"
PHOT_TEST = PROJECT_ROOT / "experiments" / "28_phot_test_features.npy"
PHOT_NAMES = PROJECT_ROOT / "experiments" / "28_phot_train_features.names.npy"
OOF_PROB = PROJECT_ROOT / "experiments" / "28_phot_oof_probabilities.npy"
TEST_PROB = PROJECT_ROOT / "experiments" / "28_phot_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "28_photometric_neighbours.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "28_photometric_neighbours.csv"


def _phot_features_for_space(
    tr_coords: np.ndarray,
    te_coords: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    priors: np.ndarray,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """OOF + test k-NN class fractions for one standardised feature space."""
    scaler = StandardScaler().fit(tr_coords)
    tr_s = scaler.transform(tr_coords)
    te_s = scaler.transform(te_coords)
    oof, names_raw = oof_neighbour_features(
        tr_s, y, fold_ids, KS_PHOT, 3, priors, SMOOTHING, MAX_K_PHOT
    )
    test, _ = neighbour_features(
        te_s, tr_s, y, KS_PHOT, 3, priors, SMOOTHING, MAX_K_PHOT
    )
    names = [f"{prefix}_{n}" for n in names_raw]
    return oof.astype(np.float32), test.astype(np.float32), names


def build_phot_features(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build or load all three photometric-neighbour feature spaces."""
    if PHOT_TRAIN.exists() and PHOT_TEST.exists() and PHOT_NAMES.exists():
        print("reusing cached photometric neighbour features")
        return (
            np.load(PHOT_TRAIN),
            np.load(PHOT_TEST),
            list(np.load(PHOT_NAMES, allow_pickle=True)),
        )

    priors = np.bincount(y, minlength=3) / len(y)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    skf = StratifiedKFold(SPATIAL_N_FOLDS, shuffle=True, random_state=SPATIAL_FOLD_SEED)
    for f, (_, va) in enumerate(skf.split(X_tr, y)):
        fold_ids[va] = f

    # Space A: colour indices
    print("building colour-space (4D) photometric neighbour features …")
    color_tr = X_tr[["u_g", "g_r", "r_i", "i_z"]].to_numpy(np.float64)
    color_te = X_te[["u_g", "g_r", "r_i", "i_z"]].to_numpy(np.float64)
    oof_a, test_a, names_a = _phot_features_for_space(
        color_tr, color_te, y, fold_ids, priors, "color"
    )
    print(f"  colour features: {oof_a.shape[1]}")

    # Space B: raw magnitudes
    print("building magnitude-space (5D) photometric neighbour features …")
    mag_tr = X_tr[["u", "g", "r", "i", "z"]].to_numpy(np.float64)
    mag_te = X_te[["u", "g", "r", "i", "z"]].to_numpy(np.float64)
    oof_b, test_b, names_b = _phot_features_for_space(
        mag_tr, mag_te, y, fold_ids, priors, "mag"
    )
    print(f"  magnitude features: {oof_b.shape[1]}")

    # Space C: sphere + colour (7D)
    print("building sphere+colour (7D) photometric neighbour features …")
    xyz_tr = radec_to_xyz(X_tr["alpha"].to_numpy(), X_tr["delta"].to_numpy())
    xyz_te = radec_to_xyz(X_te["alpha"].to_numpy(), X_te["delta"].to_numpy())
    joint_tr = np.hstack([xyz_tr, color_tr])
    joint_te = np.hstack([xyz_te, color_te])
    oof_c, test_c, names_c = _phot_features_for_space(
        joint_tr, joint_te, y, fold_ids, priors, "joint"
    )
    print(f"  sphere+colour features: {oof_c.shape[1]}")

    all_oof = np.hstack([oof_a, oof_b, oof_c])
    all_test = np.hstack([test_a, test_b, test_c])
    all_names = names_a + names_b + names_c

    np.save(PHOT_TRAIN, all_oof)
    np.save(PHOT_TEST, all_test)
    np.save(PHOT_NAMES, np.array(all_names, dtype=object))
    return all_oof, all_test, all_names


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

    if not SP_TRAIN.exists():
        raise FileNotFoundError(
            "Run scripts/15_spatial_features.py first to build position-only spatial features."
        )
    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for j, nm in enumerate(sp_names):
        X[nm] = sp_tr[:, j]
        X_test[nm] = sp_te[:, j]

    phot_tr, phot_te, phot_names = build_phot_features(X, X_test, y)
    for j, nm in enumerate(phot_names):
        X[nm] = phot_tr[:, j]
        X_test[nm] = phot_te[:, j]

    print(f"feature matrix: {X.shape} ({len(sp_names)} spatial + {len(phot_names)} phot-nbr features)")
    oof, test_prob = run_cv(X, y, X_test, cat_cols)

    argmax_score = balanced_accuracy(y, oof.argmax(1))
    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)

    np.save(OOF_PROB, oof)
    np.save(TEST_PROB, test_prob)

    print("\n================ PHOTOMETRIC NEIGHBOURS RESULT ================")
    print(f"argmax OOF                       : {argmax_score:.6f}")
    print(f"tuned  OOF                       : {tuned_score:.6f}")
    print(f"  vs incumbent {INCUMBENT_OOF:.6f}    : {tuned_score - INCUMBENT_OOF:+.6f}")
    print(f"per-class recall (tuned)         : {recalls}")
    print(f"chosen multipliers               : {mult.round(4).tolist()}")

    gate = "PASSED" if tuned_score > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "photometric_spaces": ["colour_4D", "magnitude_5D", "sphere_colour_7D"],
        "ks_phot": KS_PHOT,
        "max_k_phot": MAX_K_PHOT,
        "smoothing": SMOOTHING,
        "n_spatial_features": len(sp_names),
        "n_phot_features": len(phot_names),
        "params": LGBM_PARAMS,
        "cv_seeds": CV_SEEDS,
        "argmax_oof": argmax_score,
        "tuned_oof": tuned_score,
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": mult.tolist(),
        "per_class_recall": recalls,
    }

    if gate == "FAILED":
        print("\nFAILED ACCEPTANCE GATE — not writing submission")
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
