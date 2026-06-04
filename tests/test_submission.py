"""Regression tests for generated submission files."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.validate import validate_submission

REFERENCE = Path("raw_data/sample_submission.csv")
SUBMISSIONS_DIR = Path("submissions")

pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="raw_data/ is not present (gitignored)")


def test_generated_submissions_match_competition_format() -> None:
    submission_paths = sorted(SUBMISSIONS_DIR.glob("*.csv"))
    if not submission_paths:
        pytest.skip("no generated submissions yet")

    for submission_path in submission_paths:
        validate_submission(submission_path, REFERENCE)
