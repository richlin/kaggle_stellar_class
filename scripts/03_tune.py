"""Phase 4 LightGBM tuning, ablation reporting, and final submission generation."""
# ruff: noqa: E402
from __future__ import annotations

import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    append_jsonl,
    balanced_accuracy,
    per_class_recall,
    predict_with_multipliers,
    write_json,
)

PHASE3_RECORD_PATH = Path("experiments/02_cv_threshold.json")
EXPERIMENT_PATH = Path("experiments/03_tune.json")
RUNS_PATH = Path("experiments/runs.jsonl")
OOF_PROB_PATH = Path("experiments/03_final_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/03_final_test_probabilities.npy")
SUBMISSION_PATH = Path("submissions/03_final.csv")

N_SPLITS = 5
ABLATION_N_SPLITS = 3
SEEDS = [42, 43, 44]
MATERIAL_FOLD_REGRESSION = 0.002
MATERIAL_CLASS_RECALL_REGRESSION = 0.003
STABLE_THRESHOLD_GRID = np.array([0.75, 0.8, 0.9, 1.0, 1.1, 1.15])

BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 700,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "n_jobs": -1,
    "verbosity": -1,
}

PARAMETER_CANDIDATES: list[dict[str, Any]] = [
    {"name": "phase3_like", "params": BASE_PARAMS},
    {
        "name": "regularized_leaf47",
        "params": {
            **BASE_PARAMS,
            "n_estimators": 900,
            "learning_rate": 0.035,
            "num_leaves": 47,
            "min_child_samples": 40,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.9,
            "lambda_l1": 0.05,
            "lambda_l2": 0.2,
        },
    },
    {
        "name": "conservative_leaf31",
        "params": {
            **BASE_PARAMS,
            "n_estimators": 800,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 60,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.0,
            "lambda_l2": 0.5,
        },
    },
]

FEATURE_FAMILIES: dict[str, list[str]] = {
    "raw_magnitudes": ["u", "g", "r", "i", "z"],
    "colors": ["u_g", "g_r", "r_i", "i_z", "u_r", "u_i", "u_z", "g_i", "g_z", "r_z"],
    "magnitude_summaries": ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_range"],
    "coordinate_encoding": ["alpha", "delta", "alpha_sin", "alpha_cos"],
    "redshift_interactions": [
        "redshift_x_u_g",
        "redshift_x_g_r",
        "redshift_x_r_i",
        "redshift_x_i_z",
    ],
    "categorical_interaction": ["spectral_population"],
}


def drop_feature_family(
    X: pd.DataFrame,
    categorical_columns: list[str],
    family: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``X`` without one feature family and matching categorical columns."""
    if family not in FEATURE_FAMILIES:
        raise ValueError(f"unknown feature family: {family}")

    columns_to_drop = [column for column in FEATURE_FAMILIES[family] if column in X.columns]
    X_reduced = X.drop(columns=columns_to_drop)
    reduced_categoricals = [
        column for column in categorical_columns if column not in set(columns_to_drop)
    ]
    return X_reduced, reduced_categoricals


def select_final_candidate(
    candidates: list[dict[str, Any]],
    reference_score: float,
) -> dict[str, Any]:
    """Select the best repeated-seed candidate and annotate whether it beats Phase 3."""
    if not candidates:
        raise ValueError("at least one candidate is required")

    selected = max(candidates, key=lambda candidate: candidate["repeated_mean_chosen_oof"])
    return {
        **selected,
        "beats_reference": selected["repeated_mean_chosen_oof"] >= reference_score,
    }


def make_submission(
    ids: pd.Series,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a competition submission from averaged probabilities."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": ids.to_numpy(), "class": labels})


def load_cached_runs(path: Path = RUNS_PATH) -> list[dict[str, Any]]:
    """Load existing JSONL run records for resumable Phase 4 execution."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_cached_run(
    rows: list[dict[str, Any]],
    kind: str,
    field: str,
    value: str,
) -> dict[str, Any] | None:
    """Return the first cached run matching ``kind`` and ``field == value``."""
    for row in rows:
        if row.get("kind") == kind and row.get(field) == value:
            return row
    return None


def search_stable_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
    class_labels: list[str],
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find the best multiplier vector that improves OOF without material instability."""
    if grid is None:
        grid = STABLE_THRESHOLD_GRID

    baseline_pred = probabilities.argmax(axis=1)
    baseline_score = balanced_accuracy(y_true, baseline_pred)
    baseline_recall = per_class_recall(y_true, baseline_pred, class_labels)
    baseline_fold_scores = {
        int(fold): balanced_accuracy(y_true[fold_ids == fold], baseline_pred[fold_ids == fold])
        for fold in sorted(np.unique(fold_ids))
    }

    best: dict[str, Any] | None = None
    for values in itertools.product(grid, repeat=probabilities.shape[1]):
        multipliers = np.array(values, dtype=float)
        pred = predict_with_multipliers(probabilities, multipliers)
        score = balanced_accuracy(y_true, pred)
        if score <= baseline_score:
            continue

        recall = per_class_recall(y_true, pred, class_labels)
        class_deltas = {label: recall[label] - baseline_recall[label] for label in class_labels}
        if min(class_deltas.values()) < -MATERIAL_CLASS_RECALL_REGRESSION:
            continue

        fold_deltas = {
            int(fold): balanced_accuracy(y_true[fold_ids == fold], pred[fold_ids == fold])
            - baseline_fold_scores[int(fold)]
            for fold in sorted(np.unique(fold_ids))
        }
        if min(fold_deltas.values()) < -MATERIAL_FOLD_REGRESSION:
            continue

        candidate = {
            "accepted": True,
            "multipliers": multipliers,
            "baseline_score": baseline_score,
            "score": score,
            "class_recall_deltas": class_deltas,
            "fold_score_deltas": fold_deltas,
        }
        if best is None or score > best["score"]:
            best = candidate

    if best is not None:
        return best

    return {
        "accepted": False,
        "multipliers": np.ones(probabilities.shape[1], dtype=float),
        "baseline_score": baseline_score,
        "score": baseline_score,
        "class_recall_deltas": dict.fromkeys(class_labels, 0.0),
        "fold_score_deltas": {int(fold): 0.0 for fold in sorted(np.unique(fold_ids))},
    }


def run_cv_probabilities(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    class_labels: list[str],
    params: dict[str, Any],
    seed: int,
    n_splits: int = N_SPLITS,
) -> dict[str, Any]:
    """Train one CV run and return OOF/test probabilities plus metrics."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probabilities = np.zeros((len(X), len(class_labels)), dtype=np.float64)
    test_probabilities = np.zeros((len(X_test), len(class_labels)), dtype=np.float64)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []

    run_params = {**params, "random_state": seed}
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"  seed {seed} fold {fold}/{n_splits}")
        model = LGBMClassifier(**run_params)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            eval_metric="multi_logloss",
            categorical_feature=categorical_columns,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
        )

        valid_probabilities = model.predict_proba(X.iloc[valid_idx])
        oof_probabilities[valid_idx] = valid_probabilities
        test_probabilities += model.predict_proba(X_test) / n_splits
        fold_ids[valid_idx] = fold

        train_pred = model.predict(X.iloc[train_idx])
        valid_pred = valid_probabilities.argmax(axis=1)
        train_score = balanced_accuracy(y[train_idx], train_pred)
        valid_score = balanced_accuracy(y[valid_idx], valid_pred)
        fold_records.append(
            {
                "fold": fold,
                "best_iteration": int(model.best_iteration_ or run_params["n_estimators"]),
                "train_balanced_accuracy": train_score,
                "valid_balanced_accuracy": valid_score,
                "overfit_gap": train_score - valid_score,
                "per_class_recall": per_class_recall(y[valid_idx], valid_pred, class_labels),
            }
        )

    argmax_pred = oof_probabilities.argmax(axis=1)
    argmax_score = balanced_accuracy(y, argmax_pred)
    threshold = search_stable_multipliers(y, oof_probabilities, fold_ids, class_labels)
    chosen_pred = predict_with_multipliers(oof_probabilities, threshold["multipliers"])
    chosen_score = balanced_accuracy(y, chosen_pred)

    return {
        "seed": seed,
        "n_splits": n_splits,
        "params": run_params,
        "oof_probabilities": oof_probabilities,
        "test_probabilities": test_probabilities,
        "fold_ids": fold_ids,
        "folds": fold_records,
        "argmax_oof_balanced_accuracy": argmax_score,
        "chosen_oof_balanced_accuracy": chosen_score,
        "chosen_multipliers": threshold["multipliers"],
        "stable_threshold": threshold,
        "per_class_recall_chosen": per_class_recall(y, chosen_pred, class_labels),
        "mean_overfit_gap": float(np.mean([fold["overfit_gap"] for fold in fold_records])),
    }


def summarize_run_for_json(run: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe summary of a CV run without probability matrices."""
    return {
        key: value
        for key, value in run.items()
        if key not in {"oof_probabilities", "test_probabilities", "fold_ids"}
    } | {
        "chosen_multipliers": run["chosen_multipliers"].tolist(),
        "stable_threshold": {
            **run["stable_threshold"],
            "multipliers": run["stable_threshold"]["multipliers"].tolist(),
        },
    }


def evaluate_candidate(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    class_labels: list[str],
    seed: int,
    cached_runs: list[dict[str, Any]],
    n_splits: int = N_SPLITS,
) -> dict[str, Any]:
    """Run a single-seed candidate evaluation and return a JSON-safe summary."""
    cached = find_cached_run(cached_runs, "candidate_screen", "name", name)
    if cached is not None:
        print(f"reusing cached candidate {name}")
        return {key: value for key, value in cached.items() if key != "kind"}

    print(f"screening candidate {name}")
    run = run_cv_probabilities(X, y, X_test, categorical_columns, class_labels, params, seed, n_splits)
    summary = summarize_run_for_json(run)
    summary["name"] = name
    append_jsonl(RUNS_PATH, {"kind": "candidate_screen", **summary})
    return summary


def run_repeated_candidate(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    class_labels: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluate one candidate over multiple seeds and average probabilities."""
    runs = []
    oof_sum: np.ndarray | None = None
    test_sum: np.ndarray | None = None
    fold_ids = np.full(len(X), -1, dtype=np.int16)

    for seed in seeds:
        print(f"repeating candidate {name} with seed {seed}")
        run = run_cv_probabilities(X, y, X_test, categorical_columns, class_labels, params, seed)
        runs.append(summarize_run_for_json(run))
        oof_sum = run["oof_probabilities"] if oof_sum is None else oof_sum + run["oof_probabilities"]
        test_sum = run["test_probabilities"] if test_sum is None else test_sum + run["test_probabilities"]
        if seed == seeds[0]:
            fold_ids = run["fold_ids"]

    assert oof_sum is not None
    assert test_sum is not None
    averaged_oof = oof_sum / len(seeds)
    averaged_test = test_sum / len(seeds)
    threshold = search_stable_multipliers(y, averaged_oof, fold_ids, class_labels)
    chosen_pred = predict_with_multipliers(averaged_oof, threshold["multipliers"])
    repeated_score = balanced_accuracy(y, chosen_pred)
    repeated_summary = {
        "name": name,
        "params": params,
        "seeds": seeds,
        "runs": runs,
        "repeated_mean_chosen_oof": repeated_score,
        "per_class_recall_chosen": per_class_recall(y, chosen_pred, class_labels),
        "mean_run_score": float(np.mean([run["chosen_oof_balanced_accuracy"] for run in runs])),
        "mean_overfit_gap": float(np.mean([run["mean_overfit_gap"] for run in runs])),
        "chosen_multipliers": threshold["multipliers"].tolist(),
        "stable_threshold": {
            **threshold,
            "multipliers": threshold["multipliers"].tolist(),
        },
        "oof_probabilities": averaged_oof,
        "test_probabilities": averaged_test,
    }
    append_jsonl(
        RUNS_PATH,
        {
            "kind": "repeated_final",
            **{
                key: value
                for key, value in repeated_summary.items()
                if key not in {"oof_probabilities", "test_probabilities"}
            },
        },
    )
    return repeated_summary


def run_ablation_table(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    class_labels: list[str],
    reference_score: float,
    cached_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one compact ablation pass by dropping each feature family."""
    rows: list[dict[str, Any]] = []
    for family in FEATURE_FAMILIES:
        cached = find_cached_run(cached_runs, "feature_ablation", "dropped_family", family)
        if cached is not None:
            print(f"reusing cached ablation drop {family}")
            rows.append({key: value for key, value in cached.items() if key != "kind"})
            continue

        print(f"ablation drop {family}")
        X_reduced, reduced_categoricals = drop_feature_family(X, categorical_columns, family)
        X_test_reduced = X_test[X_reduced.columns]
        run = run_cv_probabilities(
            X_reduced,
            y,
            X_test_reduced,
            reduced_categoricals,
            class_labels,
            params,
            seed=SEEDS[0],
            n_splits=ABLATION_N_SPLITS,
        )
        row = {
            "dropped_family": family,
            "n_splits": ABLATION_N_SPLITS,
            "chosen_oof_balanced_accuracy": run["chosen_oof_balanced_accuracy"],
            "delta_vs_reference": run["chosen_oof_balanced_accuracy"] - reference_score,
            "mean_overfit_gap": run["mean_overfit_gap"],
        }
        rows.append(row)
        append_jsonl(RUNS_PATH, {"kind": "feature_ablation", **row})
    return rows


def _load_reference_score() -> float:
    record = json.loads(PHASE3_RECORD_PATH.read_text())
    return float(record["chosen_oof_balanced_accuracy"])


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    reference_score = _load_reference_score()
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    class_labels = encoder.classes_.tolist()
    cached_runs = load_cached_runs()

    candidate_summaries = [
        evaluate_candidate(
            candidate["name"],
            candidate["params"],
            X,
            y,
            X_test,
            categorical_columns,
            class_labels,
            seed=SEEDS[0],
            cached_runs=cached_runs,
        )
        for candidate in PARAMETER_CANDIDATES
    ]
    best_screened = max(candidate_summaries, key=lambda candidate: candidate["chosen_oof_balanced_accuracy"])
    best_params = next(
        candidate["params"]
        for candidate in PARAMETER_CANDIDATES
        if candidate["name"] == best_screened["name"]
    )

    ablation_table = run_ablation_table(
        best_params,
        X,
        y,
        X_test,
        categorical_columns,
        class_labels,
        reference_score,
        cached_runs,
    )

    repeated = run_repeated_candidate(
        best_screened["name"],
        best_params,
        X,
        y,
        X_test,
        categorical_columns,
        class_labels,
        SEEDS,
    )
    selected = select_final_candidate([repeated], reference_score)

    np.save(OOF_PROB_PATH, repeated["oof_probabilities"])
    np.save(TEST_PROB_PATH, repeated["test_probabilities"])
    submission = make_submission(
        sample_submission["id"],
        repeated["test_probabilities"],
        np.array(selected["chosen_multipliers"]),
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record = {
        "timestamp_utc": _timestamp(),
        "reference_phase3_oof": reference_score,
        "parameter_candidates": candidate_summaries,
        "best_screened_candidate": best_screened["name"],
        "ablation_table": ablation_table,
        "selected": {
            key: value
            for key, value in selected.items()
            if key not in {"oof_probabilities", "test_probabilities"}
        },
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "oof_probability_shape": list(repeated["oof_probabilities"].shape),
        "test_probability_shape": list(repeated["test_probabilities"].shape),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)

    print(f"reference Phase 3 OOF: {reference_score:.6f}")
    print(f"best screened candidate: {best_screened['name']}")
    print(f"final repeated-seed OOF: {selected['repeated_mean_chosen_oof']:.6f}")
    print(f"beats reference: {selected['beats_reference']}")
    print(f"chosen multipliers: {selected['chosen_multipliers']}")
    print("ablation table:")
    for row in ablation_table:
        print(
            f"  drop {row['dropped_family']}: "
            f"{row['chosen_oof_balanced_accuracy']:.6f} "
            f"(delta {row['delta_vs_reference']:.6f})"
        )
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
