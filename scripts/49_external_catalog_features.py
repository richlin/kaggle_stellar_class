"""Task 54: add guarded external-catalog numeric features to spatial model.

The catalog may join by `id` or by nearest sky coordinate. Label-like columns are
rejected and all external features are numeric, prefixed, imputed from train
medians, and paired with missing indicators.
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
from src.external_catalog import build_external_catalog_features
from src.validate import validate_submission
from src.validation import per_class_recall, search_class_multipliers, write_json

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
CV_SEEDS = [42, 43, 44]
CV_N_SPLITS = 5
INCUMBENT_OOF = 0.969211

SP_TRAIN = PROJECT_ROOT / "experiments" / "15_spatial_train_features.npy"
SP_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_features.npy"
SP_NAMES = PROJECT_ROOT / "experiments" / "15_spatial_train_features.names.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "49_external_catalog_features.json"
OOF_PROB_OUT = PROJECT_ROOT / "experiments" / "49_external_catalog_oof_probabilities.npy"
TEST_PROB_OUT = PROJECT_ROOT / "experiments" / "49_external_catalog_test_probabilities.npy"
SUBMISSION = PROJECT_ROOT / "submissions" / "49_external_catalog_features.csv"

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


def run_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_cols: list[str]):
    """Run repeated competition-only OOF CV."""
    oof = np.zeros((len(X), 3))
    test = np.zeros((len(X_test), 3))
    n_runs = len(CV_SEEDS)

    for seed in CV_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            print(f"  seed {seed} fold {fold}/{CV_N_SPLITS}")
            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(
                X.iloc[tr],
                y[tr],
                eval_set=[(X.iloc[va], y[va])],
                eval_metric="multi_logloss",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            oof[va] += model.predict_proba(X.iloc[va]) / n_runs
            test += model.predict_proba(X_test) / (n_runs * CV_N_SPLITS)

    return oof, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="External catalog CSV path")
    parser.add_argument("--join", choices=["id", "sky"], default="id")
    parser.add_argument("--max-arcsec", type=float, default=1.0)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"BLOCKED: external catalog not found: {catalog_path}")
        write_json(EXPERIMENT, {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "gate": "BLOCKED",
            "reason": f"external catalog not found: {catalog_path}",
        })
        return 0

    train, test, sample = load_raw()
    catalog = pd.read_csv(catalog_path)
    X, y, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)

    sp_tr = np.load(SP_TRAIN)
    sp_te = np.load(SP_TEST)
    sp_names = list(np.load(SP_NAMES, allow_pickle=True))
    for idx, name in enumerate(sp_names):
        X[name] = sp_tr[:, idx]
        X_test[name] = sp_te[:, idx]

    try:
        ext_train, ext_test, external_names = build_external_catalog_features(
            train,
            test,
            catalog,
            join=args.join,
            max_arcsec=args.max_arcsec,
        )
    except ValueError as exc:
        print(f"BLOCKED: {exc}")
        write_json(EXPERIMENT, {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "gate": "BLOCKED",
            "reason": str(exc),
            "catalog_path": str(catalog_path),
        })
        return 0

    for column in ext_train.columns:
        X[column] = ext_train[column]
        X_test[column] = ext_test[column]

    oof, test_prob = run_cv(X, y, X_test, cat_cols)
    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)
    np.save(OOF_PROB_OUT, oof)
    np.save(TEST_PROB_OUT, test_prob)

    gate = "PASSED" if tuned_score > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "catalog_path": str(catalog_path),
        "join": args.join,
        "max_arcsec": args.max_arcsec,
        "external_features": external_names,
        "all_added_columns": ext_train.columns.tolist(),
        "tuned_oof": tuned_score,
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": mult.tolist(),
        "per_class_recall": recalls,
        "params": LGBM_PARAMS,
        "cv_seeds": CV_SEEDS,
    }

    if gate == "FAILED":
        print(f"FAILED gate: {tuned_score:.6f} <= {INCUMBENT_OOF:.6f}")
        write_json(EXPERIMENT, record)
        return 0

    predicted = (test_prob * mult).argmax(1)
    submission = pd.DataFrame({"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(predicted)})
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)
    record["submission_path"] = str(SUBMISSION)
    write_json(EXPERIMENT, record)
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
