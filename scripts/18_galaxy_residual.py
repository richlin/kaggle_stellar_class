"""Task 24 follow-up: GALAXY-only residual correction.

The broad residual model in `17_transductive_spatial.py` did not improve the
incumbent. This script narrows the correction: train binary residual models only
where the incumbent predicts STAR or QSO, then tune thresholds for flipping those
specific rows back to GALAXY.
"""
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.metrics import confusion_matrix

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import balanced_accuracy, per_class_recall, write_json

GALAXY = 0
QSO = 1
STAR = 2
CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
INCUMBENT_MULTIPLIERS = np.array([0.6, 1.0, 1.35])

OOF_STAR_TO_GALAXY = PROJECT_ROOT / "experiments" / "18_star_to_galaxy_oof.npy"
TEST_STAR_TO_GALAXY = PROJECT_ROOT / "experiments" / "18_star_to_galaxy_test.npy"
OOF_QSO_TO_GALAXY = PROJECT_ROOT / "experiments" / "18_qso_to_galaxy_oof.npy"
TEST_QSO_TO_GALAXY = PROJECT_ROOT / "experiments" / "18_qso_to_galaxy_test.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "18_galaxy_residual.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "18_galaxy_residual.csv"

BINARY_PARAMS = {
    "objective": "binary",
    "class_weight": "balanced",
    "n_estimators": 500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 30,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "n_jobs": -1,
    "verbosity": -1,
}


def _load_task24_module():
    module_path = PROJECT_ROOT / "scripts" / "17_transductive_spatial.py"
    spec = importlib.util.spec_from_file_location("task24_transductive_spatial", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def apply_galaxy_overrides(
    incumbent_pred: np.ndarray,
    star_to_galaxy: np.ndarray,
    qso_to_galaxy: np.ndarray,
    star_threshold: float,
    qso_threshold: float,
) -> np.ndarray:
    """Flip only incumbent STAR/QSO predictions with enough GALAXY residual evidence."""
    corrected = incumbent_pred.copy()
    corrected[(incumbent_pred == STAR) & (star_to_galaxy >= star_threshold)] = GALAXY
    corrected[(incumbent_pred == QSO) & (qso_to_galaxy >= qso_threshold)] = GALAXY
    return corrected


def run_binary_residual_cv(
    X,
    y: np.ndarray,
    X_test,
    cat_cols: list[str],
    fold_ids: np.ndarray,
    incumbent_pred: np.ndarray,
    incumbent_test_pred: np.ndarray,
    target_pred: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train OOF binary model for `true GALAXY` inside one incumbent prediction region."""
    oof = np.zeros(len(y), dtype=np.float32)
    test_prob = np.zeros(len(X_test), dtype=np.float32)
    test_mask = incumbent_test_pred == target_pred
    for fold in np.unique(fold_ids):
        valid = fold_ids == fold
        train_rows = (~valid) & (incumbent_pred == target_pred)
        valid_rows = valid & (incumbent_pred == target_pred)
        binary_y = (y[train_rows] == GALAXY).astype(int)
        if len(np.unique(binary_y)) < 2:
            continue
        print(
            f"  binary target={CLASS_LABELS[target_pred]} fold {fold + 1}/{len(np.unique(fold_ids))}"
        )
        model = LGBMClassifier(**BINARY_PARAMS, random_state=4096 + target_pred * 10 + int(fold))
        model.fit(
            X.loc[train_rows],
            binary_y,
            categorical_feature=cat_cols,
            callbacks=[log_evaluation(0)],
        )
        if valid_rows.any():
            oof[valid_rows] = model.predict_proba(X.loc[valid_rows])[:, 1]
        if test_mask.any():
            test_prob[test_mask] += model.predict_proba(X_test.loc[test_mask])[:, 1] / len(
                np.unique(fold_ids)
            )
    return oof, test_prob


def search_override_thresholds(
    y: np.ndarray,
    incumbent_pred: np.ndarray,
    star_to_galaxy: np.ndarray,
    qso_to_galaxy: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Grid-search GALAXY override thresholds on OOF balanced accuracy."""
    best_star = 1.01
    best_qso = 1.01
    best_pred = incumbent_pred
    best_score = balanced_accuracy(y, incumbent_pred)
    star_grid = np.linspace(0.05, 0.95, 37)
    qso_grid = np.linspace(0.05, 0.95, 37)
    for star_threshold in star_grid:
        for qso_threshold in qso_grid:
            pred = apply_galaxy_overrides(
                incumbent_pred,
                star_to_galaxy,
                qso_to_galaxy,
                float(star_threshold),
                float(qso_threshold),
            )
            score = balanced_accuracy(y, pred)
            if score > best_score:
                best_score = score
                best_star = float(star_threshold)
                best_qso = float(qso_threshold)
                best_pred = pred
    return best_star, best_qso, best_score, best_pred


def main() -> int:
    task24 = _load_task24_module()
    train, test, sample = load_raw()
    _X_base, y, _cat, encoder = build_features(train)
    if y is None:
        raise ValueError("training data must include class labels")

    fold_ids = task24.make_fold_ids(y)
    X, X_test, cat_cols, incumbent_oof, incumbent_test = task24.build_model_frames(
        train,
        test,
        y,
        fold_ids,
    )
    incumbent_pred = (incumbent_oof * INCUMBENT_MULTIPLIERS).argmax(axis=1)
    incumbent_test_pred = (incumbent_test * INCUMBENT_MULTIPLIERS).argmax(axis=1)
    incumbent_score = balanced_accuracy(y, incumbent_pred)

    star_oof, star_test = run_binary_residual_cv(
        X,
        y,
        X_test,
        cat_cols,
        fold_ids,
        incumbent_pred,
        incumbent_test_pred,
        STAR,
    )
    qso_oof, qso_test = run_binary_residual_cv(
        X,
        y,
        X_test,
        cat_cols,
        fold_ids,
        incumbent_pred,
        incumbent_test_pred,
        QSO,
    )
    np.save(OOF_STAR_TO_GALAXY, star_oof)
    np.save(TEST_STAR_TO_GALAXY, star_test)
    np.save(OOF_QSO_TO_GALAXY, qso_oof)
    np.save(TEST_QSO_TO_GALAXY, qso_test)

    star_threshold, qso_threshold, score, pred = search_override_thresholds(
        y,
        incumbent_pred,
        star_oof,
        qso_oof,
    )
    test_pred = apply_galaxy_overrides(
        incumbent_test_pred,
        star_test,
        qso_test,
        star_threshold,
        qso_threshold,
    )
    submission = task24.make_submission(sample, np.eye(len(CLASS_LABELS))[test_pred], np.ones(3), encoder)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)

    recalls = per_class_recall(y, pred, CLASS_LABELS)
    write_json(
        EXPERIMENT,
        {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
            "incumbent_oof": incumbent_score,
            "galaxy_residual_oof": score,
            "star_threshold": star_threshold,
            "qso_threshold": qso_threshold,
            "per_class_recall": recalls,
            "n_star_to_galaxy_oof": int(((incumbent_pred == STAR) & (pred == GALAXY)).sum()),
            "n_qso_to_galaxy_oof": int(((incumbent_pred == QSO) & (pred == GALAXY)).sum()),
            "n_star_to_galaxy_test": int(
                ((incumbent_test_pred == STAR) & (test_pred == GALAXY)).sum()
            ),
            "n_qso_to_galaxy_test": int(((incumbent_test_pred == QSO) & (test_pred == GALAXY)).sum()),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "params": BINARY_PARAMS,
            "submission_path": str(SUBMISSION),
        },
    )

    print("\n================ GALAXY RESIDUAL RESULT ================")
    print(f"incumbent OOF        : {incumbent_score:.6f}")
    print(f"galaxy residual OOF  : {score:.6f}")
    print(f"thresholds STAR/QSO  : {star_threshold:.3f} / {qso_threshold:.3f}")
    print(f"per-class recall     : {recalls}")
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
