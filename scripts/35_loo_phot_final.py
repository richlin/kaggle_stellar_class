"""LOO photometric + spatial final-only candidate.

Builds leave-one-out photometric-neighbour features (colour 4D, magnitude 5D,
sphere+colour 7D) — same spaces as script 28 but LOO so train-time feature
density matches test-time, mirroring what script 19 did for spatial features.

Uses cached LOO spatial features from script 19.
No honest OOF is available (final-only candidate).

Requires: scripts/19_loo_spatial_final.py and scripts/28_photometric_neighbours.py
to have been run first (for cached LOO spatial train features).
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
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from src.data import build_features, load_raw
from src.spatial import (
    loo_neighbour_features,
    neighbour_features,
    radec_to_xyz,
)
from src.validate import validate_submission
from src.validation import write_json

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
KS_PHOT = [5, 10, 25, 50]
MAX_K_PHOT = 50
SMOOTHING = 10.0
SEEDS = [42, 43, 44, 45, 46]
SPATIAL_BLEND_WEIGHT_LGBM = 0.55
MULTIPLIERS = np.array([0.45, 0.75, 1.0])  # from public-best script 19

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

LOO_SPATIAL_TRAIN = PROJECT_ROOT / "experiments" / "19_loo_spatial_train_features.npy"
LOO_SPATIAL_NAMES = PROJECT_ROOT / "experiments" / "19_loo_spatial_feature_names.npy"
TEST_SPATIAL = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
LOO_PHOT_TRAIN = PROJECT_ROOT / "experiments" / "35_loo_phot_train_features.npy"
LOO_PHOT_TEST = PROJECT_ROOT / "experiments" / "35_loo_phot_test_features.npy"
LOO_PHOT_NAMES = PROJECT_ROOT / "experiments" / "35_loo_phot_train_features.names.npy"
LGBM_TEST_PROB = PROJECT_ROOT / "experiments" / "35_loo_phot_lgbm_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "35_loo_phot_final.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "35_loo_phot_final.csv"


def _loo_phot_space(
    tr_coords: np.ndarray,
    te_coords: np.ndarray,
    y: np.ndarray,
    priors: np.ndarray,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler = StandardScaler().fit(tr_coords)
    tr_s = scaler.transform(tr_coords)
    te_s = scaler.transform(te_coords)
    oof, names_raw = loo_neighbour_features(
        tr_s, y, KS_PHOT, 3, priors, SMOOTHING, MAX_K_PHOT
    )
    test, _ = neighbour_features(te_s, tr_s, y, KS_PHOT, 3, priors, SMOOTHING, MAX_K_PHOT)
    names = [f"{prefix}_{n}" for n in names_raw]
    return oof.astype(np.float32), test.astype(np.float32), names


def build_loo_phot_features(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if LOO_PHOT_TRAIN.exists() and LOO_PHOT_TEST.exists() and LOO_PHOT_NAMES.exists():
        print("reusing cached LOO photometric features")
        return (
            np.load(LOO_PHOT_TRAIN),
            np.load(LOO_PHOT_TEST),
            list(np.load(LOO_PHOT_NAMES, allow_pickle=True)),
        )

    priors = np.bincount(y, minlength=3) / len(y)
    color_tr = X_tr[["u_g", "g_r", "r_i", "i_z"]].to_numpy(np.float64)
    color_te = X_te[["u_g", "g_r", "r_i", "i_z"]].to_numpy(np.float64)
    mag_tr = X_tr[["u", "g", "r", "i", "z"]].to_numpy(np.float64)
    mag_te = X_te[["u", "g", "r", "i", "z"]].to_numpy(np.float64)
    xyz_tr = radec_to_xyz(X_tr["alpha"].to_numpy(), X_tr["delta"].to_numpy())
    xyz_te = radec_to_xyz(X_te["alpha"].to_numpy(), X_te["delta"].to_numpy())

    print("building LOO colour-space (4D) features …")
    oof_a, test_a, names_a = _loo_phot_space(color_tr, color_te, y, priors, "color")
    print(f"  colour: {oof_a.shape[1]} features")

    print("building LOO magnitude-space (5D) features …")
    oof_b, test_b, names_b = _loo_phot_space(mag_tr, mag_te, y, priors, "mag")
    print(f"  magnitude: {oof_b.shape[1]} features")

    print("building LOO sphere+colour (7D) features …")
    joint_tr = np.hstack([xyz_tr, color_tr])
    joint_te = np.hstack([xyz_te, color_te])
    oof_c, test_c, names_c = _loo_phot_space(joint_tr, joint_te, y, priors, "joint")
    print(f"  sphere+colour: {oof_c.shape[1]} features")

    all_oof = np.hstack([oof_a, oof_b, oof_c])
    all_test = np.hstack([test_a, test_b, test_c])
    all_names = names_a + names_b + names_c
    np.save(LOO_PHOT_TRAIN, all_oof)
    np.save(LOO_PHOT_TEST, all_test)
    np.save(LOO_PHOT_NAMES, np.array(all_names, dtype=object))
    return all_oof, all_test, all_names


def main() -> int:
    if not LOO_SPATIAL_TRAIN.exists():
        raise FileNotFoundError("Run scripts/19_loo_spatial_final.py first.")

    train, test, sample = load_raw()
    X, y, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    # add LOO spatial features
    loo_sp_feats = np.load(LOO_SPATIAL_TRAIN)
    loo_sp_names = list(np.load(LOO_SPATIAL_NAMES, allow_pickle=True))
    sp_test_feats = np.load(TEST_SPATIAL)
    for j, nm in enumerate(loo_sp_names):
        X[nm] = loo_sp_feats[:, j]
        X_test[nm] = sp_test_feats[:, j]

    # build + add LOO photometric features
    phot_tr, phot_te, phot_names = build_loo_phot_features(X, X_test, y)
    for j, nm in enumerate(phot_names):
        X[nm] = phot_tr[:, j]
        X_test[nm] = phot_te[:, j]

    print(f"feature matrix: {X.shape}")

    # train 5-seed full LightGBM
    if LGBM_TEST_PROB.exists():
        print("reusing cached LOO phot LightGBM test probabilities")
        lgbm_test = np.load(LGBM_TEST_PROB)
    else:
        lgbm_test = np.zeros((len(X_test), len(CLASS_LABELS)))
        for seed in SEEDS:
            print(f"  full LOO phot LightGBM seed {seed}")
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(X, y, categorical_feature=cat_cols)
            lgbm_test += model.predict_proba(X_test) / len(SEEDS)
        np.save(LGBM_TEST_PROB, lgbm_test)

    xgb_test = np.load(XGB_TEST)
    blend_test = SPATIAL_BLEND_WEIGHT_LGBM * lgbm_test + (1 - SPATIAL_BLEND_WEIGHT_LGBM) * xgb_test
    predicted = (blend_test * MULTIPLIERS).argmax(1)

    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(predicted)}
    )
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)

    # class counts diagnostic
    counts = {c: int((encoder.inverse_transform(predicted) == c).sum()) for c in CLASS_LABELS}
    print(f"\nClass counts: {counts}")
    print("(reference band: GALAXY 156450–156650, QSO 51250–51450, STAR 39400–39600)")

    write_json(EXPERIMENT, {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rationale": "LOO photometric + spatial final-only candidate; no honest OOF",
        "loo_phot_spaces": ["colour_4D", "magnitude_5D", "sphere_colour_7D"],
        "seeds": SEEDS,
        "multipliers": MULTIPLIERS.tolist(),
        "blend_weight_lgbm": SPATIAL_BLEND_WEIGHT_LGBM,
        "class_counts": counts,
        "submission_path": str(SUBMISSION),
    })
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
