"""Validation, threshold tuning, and experiment logging helpers."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return balanced accuracy as a plain ``float``."""
    return float(balanced_accuracy_score(y_true, y_pred))


def per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> dict[str, float]:
    """Return recall per encoded class label."""
    labels = np.arange(len(class_labels))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: float(score) for label, score in zip(class_labels, recalls, strict=True)}


def apply_class_multipliers(probabilities: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Apply per-class multipliers without changing the probability matrix shape."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if multipliers.shape != (probabilities.shape[1],):
        raise ValueError(
            f"multipliers shape must be {(probabilities.shape[1],)}, got {multipliers.shape}"
        )
    return probabilities * multipliers


def predict_with_multipliers(probabilities: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Predict encoded classes after applying per-class multipliers."""
    return apply_class_multipliers(probabilities, multipliers).argmax(axis=1)


def search_class_multipliers(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray | None = None,
    max_rounds: int = 4,
) -> tuple[np.ndarray, float]:
    """Coordinate-ascent search for per-class multipliers maximizing balanced accuracy."""
    if grid is None:
        grid = np.linspace(0.6, 1.6, 21)

    multipliers = np.ones(probabilities.shape[1], dtype=float)
    best_score = balanced_accuracy(y_true, predict_with_multipliers(probabilities, multipliers))

    for _round in range(max_rounds):
        improved = False
        for class_idx in range(probabilities.shape[1]):
            best_value = multipliers[class_idx]
            for value in grid:
                candidate = multipliers.copy()
                candidate[class_idx] = float(value)
                score = balanced_accuracy(y_true, predict_with_multipliers(probabilities, candidate))
                if score > best_score:
                    best_score = score
                    best_value = float(value)
                    improved = True
            multipliers[class_idx] = best_value
        if not improved:
            break

    return multipliers, best_score


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    """Write an experiment record with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
