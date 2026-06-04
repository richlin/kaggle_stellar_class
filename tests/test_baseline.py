"""Focused tests for the Phase 1 baseline script helpers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import make_label_encoder


def _load_baseline_module():
    module_path = Path("scripts/01_baseline.py")
    spec = importlib.util.spec_from_file_location("baseline_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_submission_decodes_labels_and_preserves_id_order() -> None:
    baseline = _load_baseline_module()
    encoder = make_label_encoder()

    submission = baseline.make_submission(pd.Series([577347, 577348]), np.array([2, 0]), encoder)

    assert submission.columns.tolist() == ["id", "class"]
    assert submission["id"].tolist() == [577347, 577348]
    assert submission["class"].tolist() == ["STAR", "GALAXY"]


def test_write_experiment_record_writes_required_keys(tmp_path: Path) -> None:
    baseline = _load_baseline_module()
    output_path = tmp_path / "experiment.json"
    record = {
        "params": {"n_estimators": 10},
        "feature_columns": ["redshift"],
        "seed": 42,
        "holdout_balanced_accuracy": 0.95,
        "per_class_recall": {"GALAXY": 0.9, "QSO": 0.96, "STAR": 0.99},
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "submission_path": "submissions/01_baseline.csv",
        "timestamp_utc": "2026-06-03T00:00:00Z",
    }

    baseline.write_experiment_record(output_path, record)

    saved = json.loads(output_path.read_text())
    assert set(record).issubset(saved)
    assert saved["submission_path"] == "submissions/01_baseline.csv"
