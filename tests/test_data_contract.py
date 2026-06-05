"""Data-contract checks for the Kaggle stellar-class CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path("raw_data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

pytestmark = pytest.mark.skipif(not TRAIN_PATH.exists(), reason="raw_data/ is not present (gitignored)")

TRAIN_ROWS = 577_347
TEST_ROWS = 247_435
LABELS = {"GALAXY", "QSO", "STAR"}
REQUIRED_TRAIN_COLUMNS = {
    "id",
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
    "spectral_type",
    "galaxy_population",
    "class",
}
REQUIRED_TEST_COLUMNS = REQUIRED_TRAIN_COLUMNS - {"class"}
CATEGORICAL_COLUMNS = ["spectral_type", "galaxy_population"]


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def test_row_counts_match_competition_contract(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame
) -> None:
    assert len(train_df) == TRAIN_ROWS
    assert len(test_df) == TEST_ROWS
    assert len(sample_df) == TEST_ROWS


def test_required_columns_and_labels(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    assert REQUIRED_TRAIN_COLUMNS.issubset(train_df.columns)
    assert REQUIRED_TEST_COLUMNS.issubset(test_df.columns)
    assert set(train_df["class"].unique()) == LABELS


def test_ids_are_unique_and_sample_matches_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame
) -> None:
    assert train_df["id"].is_unique
    assert test_df["id"].is_unique
    assert sample_df["id"].tolist() == test_df["id"].tolist()


def test_no_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    assert not train_df.isna().any().any()
    assert not test_df.isna().any().any()


def test_test_categorical_levels_are_seen_in_train(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    for column in CATEGORICAL_COLUMNS:
        assert set(test_df[column]).issubset(set(train_df[column]))


def test_synthetic_categorical_columns_match_documented_formulae(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    combined = pd.concat([train_df, test_df], ignore_index=True)

    spectral_type = pd.cut(
        combined["r"] - combined["g"],
        [-float("inf"), -1, -0.5, 0, float("inf")],
        labels=["M", "G/K", "A/F", "O/B"],
    ).astype(str)
    galaxy_population = pd.cut(
        combined["u"] - combined["r"],
        [-float("inf"), 2.2, float("inf")],
        labels=["Blue_Cloud", "Red_Sequence"],
    ).astype(str)

    assert spectral_type.equals(combined["spectral_type"])
    assert galaxy_population.equals(combined["galaxy_population"])
