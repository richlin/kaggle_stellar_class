"""Phase 6 XGBoost hyperparameter tuning with early stopping."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    append_jsonl,
    balanced_accuracy,
    per_class_recall,
    predict_with_multipliers,
    write_json,
)

REFERENCE_SCORE = 0.9659249816190973
N_CLASSES = 3
DEFAULT_N_TRIALS = 20
SCREEN_N_SPLITS = 3
FULL_N_SPLITS = 5
SCREEN_SEED = 42
FULL_SEEDS = [52, 53, 54]

RUNS_PATH = Path("experiments/runs.jsonl")
EXPERIMENT_PATH = Path("experiments/05_tune_xgb.json")
SUBMISSION_PATH = Path("submissions/05_tuned_ensemble.csv")
TUNED_XGB_OOF_PATH = Path("experiments/05_tuned_xgb_oof_probabilities.npy")
TUNED_XGB_TEST_PATH = Path("experiments/05_tuned_xgb_test_probabilities.npy")

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
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


def _load_ensemble_module():
    spec = importlib.util.spec_from_file_location("ensemble_script", Path("scripts/04_ensemble.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def xgboost_frames(X: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical columns consistently for XGBoost."""
    combined = pd.concat([X, X_test], axis=0, ignore_index=True)
    categorical_columns = combined.select_dtypes(exclude=["number", "bool"]).columns
    encoded = pd.get_dummies(combined, columns=categorical_columns)
    return encoded.iloc[: len(X)].astype(float), encoded.iloc[len(X) :].astype(float)


def sample_xgb_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Sample one XGBoost parameter set from the Phase 6 search space."""
    return {
        **DEFAULT_XGB_PARAMS,
        "n_estimators": trial.suggest_int("n_estimators", 1500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 4.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }


def evaluate_xgb_params(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, Any],
    n_splits: int = SCREEN_N_SPLITS,
    seed: int = SCREEN_SEED,
) -> float:
    """Return CV balanced accuracy for one XGBoost parameter set."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        model = XGBClassifier(**{**params, "random_state": seed + fold})
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            sample_weight=compute_sample_weight("balanced", y[train_idx]),
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            verbose=False,
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])
    return float(balanced_accuracy_score(y, oof.argmax(axis=1)))


def make_tuned_xgb_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def run_xgb_cv_probabilities(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    n_splits: int = FULL_N_SPLITS,
) -> dict[str, Any]:
    """Train XGBoost CV probabilities for one seed."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"xgb tuned seed {seed} fold {fold}/{n_splits}")
        model = XGBClassifier(**{**params, "random_state": seed + fold})
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            sample_weight=compute_sample_weight("balanced", y[train_idx]),
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            verbose=False,
        )
        valid_probabilities = model.predict_proba(X.iloc[valid_idx])
        oof[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test) / n_splits
        fold_ids[valid_idx] = fold
        folds.append(
            {
                "fold": fold,
                "best_iteration": int(getattr(model, "best_iteration", params["n_estimators"])),
                "valid_balanced_accuracy": balanced_accuracy(y[valid_idx], valid_probabilities.argmax(axis=1)),
            }
        )
    return {
        "seed": seed,
        "oof_probabilities": oof,
        "test_probabilities": test_probabilities,
        "fold_ids": fold_ids,
        "folds": folds,
        "argmax_oof_balanced_accuracy": balanced_accuracy(y, oof.argmax(axis=1)),
    }


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def tune_and_blend(n_trials: int, validation_seeds: list[int]) -> dict[str, Any]:
    """Run Optuna screening, validate winner, and blend with reference LightGBM."""
    train_df, test_df, sample_submission = load_raw()
    X, y, _categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()
    X_xgb, X_test_xgb = xgboost_frames(X, X_test)

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_xgb_params(trial)
        return evaluate_xgb_params(X_xgb, y, params, n_splits=SCREEN_N_SPLITS, seed=SCREEN_SEED)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SCREEN_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = sample_xgb_params(study.best_trial)

    runs = [
        run_xgb_cv_probabilities(X_xgb, y, X_test_xgb, best_params, seed=seed)
        for seed in validation_seeds
    ]
    averaged_oof = sum(run["oof_probabilities"] for run in runs) / len(runs)
    averaged_test = sum(run["test_probabilities"] for run in runs) / len(runs)
    fold_ids = runs[0]["fold_ids"]
    np.save(TUNED_XGB_OOF_PATH, averaged_oof)
    np.save(TUNED_XGB_TEST_PATH, averaged_test)

    ensemble = _load_ensemble_module()
    reference_oof = np.load("experiments/03_final_oof_probabilities.npy")
    reference_test = np.load("experiments/03_final_test_probabilities.npy")
    blend = ensemble.search_blend_weights(
        y,
        [reference_oof, averaged_oof],
        fold_ids,
        class_labels,
    )
    blended_test = ensemble.weighted_probability_blend([reference_test, averaged_test], blend["weights"])
    submission = make_tuned_xgb_submission(
        sample_submission,
        blended_test,
        blend["multipliers"],
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    chosen_pred = predict_with_multipliers(
        ensemble.weighted_probability_blend([reference_oof, averaged_oof], blend["weights"]),
        blend["multipliers"],
    )
    record = {
        "timestamp_utc": _timestamp(),
        "candidate": "tuned_xgb_ensemble",
        "n_trials": n_trials,
        "screen_best_score": float(study.best_value),
        "best_params": best_params,
        "validation_seeds": validation_seeds,
        "tuned_xgb_argmax_oof": balanced_accuracy(y, averaged_oof.argmax(axis=1)),
        "blend_weights": blend["weights"].tolist(),
        "chosen_multipliers": blend["multipliers"].tolist(),
        "chosen_oof_balanced_accuracy": blend["score"],
        "delta_vs_reference": blend["score"] - REFERENCE_SCORE,
        "per_class_recall": per_class_recall(y, chosen_pred, class_labels),
        "runs": [
            {
                "seed": run["seed"],
                "argmax_oof_balanced_accuracy": run["argmax_oof_balanced_accuracy"],
                "folds": run["folds"],
            }
            for run in runs
        ],
        "oof_probability_path": str(TUNED_XGB_OOF_PATH),
        "test_probability_path": str(TUNED_XGB_TEST_PATH),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "tuned_xgb_ensemble", **record})
    return record


def _parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--validation-seeds", default="52,53,54")
    args = parser.parse_args(argv)
    record = tune_and_blend(args.trials, _parse_seeds(args.validation_seeds))
    print(f"screen best: {record['screen_best_score']:.6f}")
    print(f"tuned XGB argmax OOF: {record['tuned_xgb_argmax_oof']:.6f}")
    print(f"blend OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs reference: {record['delta_vs_reference']:.6f}")
    print(f"weights: {record['blend_weights']}")
    print(f"multipliers: {record['chosen_multipliers']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
