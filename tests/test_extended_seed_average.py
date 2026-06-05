"""Focused tests for extended seed averaging helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _load_seed_module():
    module_path = Path("scripts/09_extended_seed_average.py")
    spec = importlib.util.spec_from_file_location("extended_seed_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_combine_existing_average_with_new_runs() -> None:
    seed_average = _load_seed_module()
    existing_average = np.array([[0.6, 0.3, 0.1]])
    new_runs = [np.array([[0.3, 0.6, 0.1]]), np.array([[0.9, 0.05, 0.05]])]

    combined = seed_average.combine_existing_average_with_new_runs(
        existing_average,
        existing_run_count=3,
        new_runs=new_runs,
    )

    expected = ((existing_average * 3) + new_runs[0] + new_runs[1]) / 5
    np.testing.assert_allclose(combined, expected)


def test_make_extended_seed_submission_preserves_id_order() -> None:
    seed_average = _load_seed_module()
    encoder = LabelEncoder().fit(["GALAXY", "QSO", "STAR"])
    sample_submission = pd.DataFrame({"id": [11, 9, 10], "class": ["GALAXY", "GALAXY", "GALAXY"]})
    probabilities = np.array(
        [
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
            [0.8, 0.1, 0.1],
        ]
    )

    submission = seed_average.make_extended_seed_submission(
        sample_submission,
        probabilities,
        np.ones(3),
        encoder,
    )

    assert submission["id"].tolist() == [11, 9, 10]
    assert submission["class"].tolist() == ["QSO", "STAR", "GALAXY"]
