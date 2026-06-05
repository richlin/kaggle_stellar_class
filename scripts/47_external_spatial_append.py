"""Task 52: append audited original labels as spatial references and weighted rows.

This experiment is the main local-OOF >0.971 candidate. It uses audited original
rows in two ways:

1. As extra labelled neighbours when computing fold-safe OOF spatial features.
2. As appended training rows with a sweep of source weights.

Validation is always scored only on competition rows.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
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
from src.external_spatial import (
    external_reference_oof_features,
    external_reference_test_features,
    make_append_sample_weights,
)
from src.spatial import neighbour_features, radec_to_xyz
from src.validate import validate_submission
from src.validation import per_class_recall, search_class_multipliers, write_json

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
CV_SEEDS = [42, 43, 44]
CV_N_SPLITS = 5
SOURCE_WEIGHTS = [0.05, 0.10, 0.25, 0.50, 1.00]
INCUMBENT_OOF = 0.969211
SPATIAL_KS = [5, 10, 25, 50, 100, 250]
SPATIAL_MAX_K = 250
SPATIAL_SMOOTHING = 10.0

EXPERIMENT = PROJECT_ROOT / "experiments" / "47_external_spatial_append.json"
OOF_PROB_OUT = PROJECT_ROOT / "experiments" / "47_external_spatial_append_oof_probabilities.npy"
TEST_PROB_OUT = PROJECT_ROOT / "experiments" / "47_external_spatial_append_test_probabilities.npy"
SUBMISSION = PROJECT_ROOT / "submissions" / "47_external_spatial_append.csv"

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


def _load_append_helpers():
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "44_original_append_train.py"
    spec = importlib.util.spec_from_file_location("original_append_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_and_prepare_original, module.verify_audit_matches_original


def _load_passed_audit(original_path: str) -> dict:
    audit_path = PROJECT_ROOT / "experiments" / "43_original_append_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("Run scripts/43_original_append_audit.py before this experiment")
    audit = json.loads(audit_path.read_text())
    if audit.get("verdict") != "PASS":
        raise ValueError(f"original append audit is {audit.get('verdict')}, not PASS")
    _load_append_helpers()[1](audit, original_path)
    return audit


def add_external_reference_spatial_features(
    X_comp: pd.DataFrame,
    X_test: pd.DataFrame,
    X_orig: pd.DataFrame,
    y_comp: np.ndarray,
    y_orig: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Add external-reference spatial features to competition, test, and original rows."""
    priors = np.bincount(np.concatenate([y_comp, y_orig]), minlength=3)
    priors = priors / priors.sum()

    comp_xyz = radec_to_xyz(X_comp["alpha"].to_numpy(), X_comp["delta"].to_numpy())
    test_xyz = radec_to_xyz(X_test["alpha"].to_numpy(), X_test["delta"].to_numpy())
    orig_xyz = radec_to_xyz(X_orig["alpha"].to_numpy(), X_orig["delta"].to_numpy())

    comp_sp, names = external_reference_oof_features(
        comp_xyz,
        y_comp,
        fold_ids,
        orig_xyz,
        y_orig,
        SPATIAL_KS,
        3,
        priors,
        SPATIAL_SMOOTHING,
        SPATIAL_MAX_K,
    )
    test_sp, _ = external_reference_test_features(
        test_xyz,
        comp_xyz,
        y_comp,
        orig_xyz,
        y_orig,
        SPATIAL_KS,
        3,
        priors,
        SPATIAL_SMOOTHING,
        SPATIAL_MAX_K,
    )
    orig_sp, _ = neighbour_features(
        orig_xyz,
        comp_xyz,
        y_comp,
        SPATIAL_KS,
        3,
        priors,
        SPATIAL_SMOOTHING,
        SPATIAL_MAX_K,
    )

    feature_names = [f"extref_{name}" for name in names]
    for idx, name in enumerate(feature_names):
        X_comp[name] = comp_sp[:, idx]
        X_test[name] = test_sp[:, idx]
        X_orig[name] = orig_sp[:, idx]
    return X_comp, X_test, X_orig, feature_names


def run_weighted_cv(
    X_comp: pd.DataFrame,
    y_comp: np.ndarray,
    X_orig: pd.DataFrame,
    y_orig: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    external_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Train with original rows appended to every fold using a source weight."""
    oof = np.zeros((len(X_comp), 3))
    test = np.zeros((len(X_test), 3))
    n_runs = len(CV_SEEDS)

    for seed in CV_SEEDS:
        skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X_comp, y_comp), 1):
            print(f"  weight {external_weight} seed {seed} fold {fold}/{CV_N_SPLITS}")
            X_train = pd.concat([X_comp.iloc[tr], X_orig], ignore_index=True)
            y_train = np.concatenate([y_comp[tr], y_orig])
            weights = make_append_sample_weights(len(tr), len(X_orig), external_weight)

            model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(
                X_train,
                y_train,
                sample_weight=weights,
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

    try:
        audit = _load_passed_audit(args.original)
    except (FileNotFoundError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        write_json(EXPERIMENT, {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "gate": "BLOCKED",
            "reason": str(exc),
            "original_path": args.original,
        })
        return 0

    load_original, _verify = _load_append_helpers()
    train, test, sample = load_raw()
    X_comp, y_comp, cat_cols, encoder = build_features(train)
    X_test, _yt, _ct, _enc = build_features(test, label_encoder=encoder)
    X_orig, y_orig = load_original(args.original, encoder)

    primary_fold_ids = np.full(len(y_comp), -1, dtype=int)
    for fold, (_tr, va) in enumerate(
        StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=CV_SEEDS[0]).split(X_comp, y_comp)
    ):
        primary_fold_ids[va] = fold

    X_comp, X_test, X_orig, spatial_names = add_external_reference_spatial_features(
        X_comp, X_test, X_orig, y_comp, y_orig, primary_fold_ids
    )

    results = []
    best = None
    for weight in SOURCE_WEIGHTS:
        oof, test_prob = run_weighted_cv(X_comp, y_comp, X_orig, y_orig, X_test, cat_cols, weight)
        mult, score = search_class_multipliers(y_comp, oof)
        recalls = per_class_recall(y_comp, (oof * mult).argmax(1), CLASS_LABELS)
        candidate = {
            "external_weight": weight,
            "oof": oof,
            "test_prob": test_prob,
            "multipliers": mult,
            "score": score,
            "per_class_recall": recalls,
        }
        results.append({
            "external_weight": weight,
            "tuned_oof": score,
            "multipliers": mult.tolist(),
            "per_class_recall": recalls,
        })
        if best is None or score > best["score"]:
            best = candidate

    assert best is not None
    np.save(OOF_PROB_OUT, best["oof"])
    np.save(TEST_PROB_OUT, best["test_prob"])

    gate = "PASSED" if best["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "audit_clean_append_rows": audit.get("clean_append_rows"),
        "original_path": args.original,
        "external_reference_spatial_features": spatial_names,
        "source_weight_results": results,
        "best_external_weight": best["external_weight"],
        "best_tuned_oof": best["score"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "params": LGBM_PARAMS,
        "cv_seeds": CV_SEEDS,
    }

    if gate == "FAILED":
        print(f"FAILED gate: {best['score']:.6f} <= {INCUMBENT_OOF:.6f}")
        write_json(EXPERIMENT, record)
        return 0

    predicted = (best["test_prob"] * best["multipliers"]).argmax(1)
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
