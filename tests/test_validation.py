"""Tests for CV metrics, threshold tuning, and experiment logging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.validation import (
    append_jsonl,
    apply_class_multipliers,
    balanced_accuracy,
    per_class_recall,
    search_class_multipliers,
    write_json,
)


def test_balanced_accuracy_matches_sklearn() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    assert balanced_accuracy(y_true, y_pred) == balanced_accuracy_score(y_true, y_pred)


def test_per_class_recall_uses_named_classes() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    recalls = per_class_recall(y_true, y_pred, ["GALAXY", "QSO", "STAR"])

    assert recalls == {"GALAXY": 0.5, "QSO": 1.0, "STAR": 0.5}


def test_apply_class_multipliers_preserves_probability_shape() -> None:
    probabilities = np.array([[0.4, 0.3, 0.3], [0.2, 0.7, 0.1]])
    multipliers = np.array([1.0, 0.5, 2.0])

    weighted = apply_class_multipliers(probabilities, multipliers)

    assert weighted.shape == probabilities.shape
    assert weighted.tolist() == [[0.4, 0.15, 0.6], [0.2, 0.35, 0.2]]


def test_threshold_search_improves_or_preserves_balanced_accuracy() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.45, 0.40, 0.15],
            [0.43, 0.42, 0.15],
            [0.50, 0.40, 0.10],
            [0.48, 0.45, 0.07],
            [0.55, 0.10, 0.35],
            [0.52, 0.08, 0.40],
        ]
    )
    baseline = balanced_accuracy(y_true, probabilities.argmax(axis=1))

    multipliers, tuned_score = search_class_multipliers(
        y_true,
        probabilities,
        grid=np.array([0.5, 1.0, 1.5, 2.0]),
        max_rounds=3,
    )

    assert multipliers.shape == (3,)
    assert tuned_score >= baseline


def test_write_json_creates_parent_and_round_trips(tmp_path: Path) -> None:
    output_path = tmp_path / "experiments" / "run.json"

    write_json(output_path, {"score": 0.97, "weights": [1.0, 0.9, 1.1]})

    assert json.loads(output_path.read_text()) == {"score": 0.97, "weights": [1.0, 0.9, 1.1]}


def test_append_jsonl_appends_one_json_record_per_line(tmp_path: Path) -> None:
    output_path = tmp_path / "experiments" / "runs.jsonl"

    append_jsonl(output_path, {"name": "candidate_a", "score": 0.96})
    append_jsonl(output_path, {"name": "candidate_b", "score": 0.97})

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {"name": "candidate_a", "score": 0.96},
        {"name": "candidate_b", "score": 0.97},
    ]
