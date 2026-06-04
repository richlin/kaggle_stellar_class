"""Smoke tests for the submission-format validator.

The real sample_submission.csv must validate; common corruptions must be caught.
Skips gracefully if raw_data/ (gitignored) is not present.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.validate import validate_submission

REFERENCE = Path("raw_data/sample_submission.csv")
pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(), reason="raw_data/sample_submission.csv not present (gitignored)"
)


def test_sample_submission_is_valid():
    # The reference validated against itself must pass.
    validate_submission(REFERENCE, REFERENCE)


def test_wrong_columns_rejected():
    bad = pd.DataFrame({"id": [577347], "label": ["STAR"]})
    with pytest.raises(ValueError, match="columns"):
        validate_submission(bad, REFERENCE)


def test_bad_label_rejected():
    ref = pd.read_csv(REFERENCE)
    bad = ref.copy()
    bad.loc[0, "class"] = "PLANET"
    with pytest.raises(ValueError, match="unexpected labels"):
        validate_submission(bad, REFERENCE)


def test_wrong_row_count_rejected():
    ref = pd.read_csv(REFERENCE)
    with pytest.raises(ValueError, match="rows"):
        validate_submission(ref.iloc[:-1], REFERENCE)


def test_mismatched_ids_rejected():
    ref = pd.read_csv(REFERENCE)
    bad = ref.copy()
    bad.loc[0, "id"] = -1  # break id alignment without changing row count
    with pytest.raises(ValueError, match="ids"):
        validate_submission(bad, REFERENCE)
