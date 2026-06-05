"""Extend the public-best LightGBM seed average from 3 seeds to 5 seeds."""
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
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

EXISTING_RUN_COUNT = 3
NEW_SEEDS = [45, 46]
REFERENCE_SCORE = 0.9659249816190973

EXISTING_OOF_PATH = Path("experiments/03_final_oof_probabilities.npy")
EXISTING_TEST_PATH = Path("experiments/03_final_test_probabilities.npy")
EXPERIMENT_PATH = Path("experiments/09_extended_seed_average.json")
OOF_PROB_PATH = Path("experiments/09_extended_seed_average_oof_probabilities.npy")
TEST_PROB_PATH = Path("experiments/09_extended_seed_average_test_probabilities.npy")
SUBMISSION_PATH = Path("submissions/09_extended_seed_average.csv")
RUNS_PATH = Path("experiments/runs.jsonl")


def _load_tune_module():
    spec = importlib.util.spec_from_file_location("tune_script", Path("scripts/03_tune.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def combine_existing_average_with_new_runs(
    existing_average: np.ndarray,
    existing_run_count: int,
    new_runs: list[np.ndarray],
) -> np.ndarray:
    """Combine an existing probability average with additional run probabilities."""
    if not new_runs:
        raise ValueError("at least one new run is required")
    for run in new_runs:
        if run.shape != existing_average.shape:
            raise ValueError("all probability matrices must have matching shape")
    total = existing_average * existing_run_count
    for run in new_runs:
        total += run
    return total / (existing_run_count + len(new_runs))


def make_extended_seed_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a submission while preserving sample-submission id order."""
    encoded_predictions = predict_with_multipliers(probabilities, multipliers)
    labels = encoder.inverse_transform(encoded_predictions)
    return pd.DataFrame({"id": sample_submission["id"].to_numpy(), "class": labels})


def run_extended_seed_average() -> dict[str, Any]:
    """Train new seeds, average with existing probabilities, and write a submission."""
    tune = _load_tune_module()
    train_df, test_df, sample_submission = load_raw()
    X, y, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)
    if y is None:
        raise ValueError("training labels are required")
    class_labels = encoder.classes_.tolist()

    new_oof_runs = []
    new_test_runs = []
    new_run_summaries = []
    fold_ids = None
    for seed in NEW_SEEDS:
        print(f"extended seed {seed}")
        run = tune.run_cv_probabilities(
            X,
            y,
            X_test,
            categorical_columns,
            class_labels,
            tune.BASE_PARAMS,
            seed=seed,
        )
        new_oof_runs.append(run["oof_probabilities"])
        new_test_runs.append(run["test_probabilities"])
        new_run_summaries.append(tune.summarize_run_for_json(run))
        fold_ids = run["fold_ids"]

    existing_oof = np.load(EXISTING_OOF_PATH)
    existing_test = np.load(EXISTING_TEST_PATH)
    averaged_oof = combine_existing_average_with_new_runs(
        existing_oof,
        EXISTING_RUN_COUNT,
        new_oof_runs,
    )
    averaged_test = combine_existing_average_with_new_runs(
        existing_test,
        EXISTING_RUN_COUNT,
        new_test_runs,
    )
    if fold_ids is None:
        raise ValueError("fold ids were not generated")

    threshold = tune.search_stable_multipliers(y, averaged_oof, fold_ids, class_labels)
    chosen_pred = predict_with_multipliers(averaged_oof, threshold["multipliers"])
    chosen_score = balanced_accuracy(y, chosen_pred)
    selection = {
        "accepted": chosen_score > REFERENCE_SCORE,
        "reference_score": REFERENCE_SCORE,
        "candidate_score": chosen_score,
        "delta": chosen_score - REFERENCE_SCORE,
    }

    np.save(OOF_PROB_PATH, averaged_oof)
    np.save(TEST_PROB_PATH, averaged_test)
    submission = make_extended_seed_submission(
        sample_submission,
        averaged_test,
        threshold["multipliers"],
        encoder,
    )
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    validate_submission(SUBMISSION_PATH, sample_submission)

    record = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate": "extended_seed_average",
        "existing_run_count": EXISTING_RUN_COUNT,
        "new_seeds": NEW_SEEDS,
        "reference_oof_balanced_accuracy": REFERENCE_SCORE,
        "chosen_oof_balanced_accuracy": chosen_score,
        "candidate_selection": selection,
        "chosen_multipliers": threshold["multipliers"].tolist(),
        "per_class_recall": per_class_recall(y, chosen_pred, class_labels),
        "stable_threshold": {
            **threshold,
            "multipliers": threshold["multipliers"].tolist(),
        },
        "new_runs": new_run_summaries,
        "oof_probability_path": str(OOF_PROB_PATH),
        "test_probability_path": str(TEST_PROB_PATH),
        "submission_path": str(SUBMISSION_PATH),
    }
    write_json(EXPERIMENT_PATH, record)
    append_jsonl(RUNS_PATH, {"kind": "extended_seed_average", **record})
    return record


def main() -> int:
    record = run_extended_seed_average()
    print(f"extended seed OOF: {record['chosen_oof_balanced_accuracy']:.6f}")
    print(f"delta vs 03_final: {record['candidate_selection']['delta']:.6f}")
    print(f"accepted: {record['candidate_selection']['accepted']}")
    print(f"multipliers: {record['chosen_multipliers']}")
    print(f"wrote {SUBMISSION_PATH}")
    print(f"wrote {EXPERIMENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
