"""Submission format validator for the stellar-class competition.

A submission is valid iff it has exactly the columns ``id,class``, one row per
test object (ids aligned to the reference), and every label in the allowed set.
Used both as a smoke test (``tests/test_validate.py``) and as a CLI guard before
uploading: ``python -m src.validate submissions/01_baseline.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ALLOWED_LABELS = {"GALAXY", "QSO", "STAR"}
EXPECTED_COLUMNS = ["id", "class"]
DEFAULT_REFERENCE = Path("raw_data/sample_submission.csv")


def validate_submission(
    submission: pd.DataFrame | str | Path,
    reference: pd.DataFrame | str | Path = DEFAULT_REFERENCE,
) -> None:
    """Raise ``ValueError`` if ``submission`` is not a valid submission.

    Accepts either a DataFrame or a path to a CSV for both arguments.
    """
    sub = submission if isinstance(submission, pd.DataFrame) else pd.read_csv(submission)
    ref = reference if isinstance(reference, pd.DataFrame) else pd.read_csv(reference)

    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"columns must be {EXPECTED_COLUMNS}, got {list(sub.columns)}")

    if len(sub) != len(ref):
        raise ValueError(f"expected {len(ref)} rows, got {len(sub)}")

    bad_labels = set(sub["class"].unique()) - ALLOWED_LABELS
    if bad_labels:
        raise ValueError(f"unexpected labels: {sorted(bad_labels)}; allowed {sorted(ALLOWED_LABELS)}")

    if set(sub["id"]) != set(ref["id"]):
        raise ValueError("submission ids do not match the reference id set")

    if sub["id"].duplicated().any():
        raise ValueError("submission contains duplicate ids")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m src.validate <submission.csv> [reference.csv]", file=sys.stderr)
        return 2
    ref = argv[1] if len(argv) > 1 else DEFAULT_REFERENCE
    try:
        validate_submission(argv[0], ref)
    except (ValueError, FileNotFoundError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {argv[0]} is a valid submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
