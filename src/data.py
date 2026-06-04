"""Shared data loading and feature/target preparation."""
# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features import build_feature_frame

RAW_DATA_DIR = Path("raw_data")
CLASS_LABELS = np.array(["GALAXY", "QSO", "STAR"])
TARGET_COLUMN = "class"


def load_raw(data_dir: str | Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample-submission CSVs from ``data_dir``."""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    return train, test, sample_submission


def make_label_encoder() -> LabelEncoder:
    """Create the competition label encoder with a stable class order."""
    encoder = LabelEncoder()
    encoder.fit(CLASS_LABELS)
    return encoder


def build_features(
    df: pd.DataFrame,
    feature_set: str = "baseline",
    label_encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, list[str], LabelEncoder]:
    """Build features and encode ``class`` when present."""
    X, categorical_columns = build_feature_frame(df, feature_set=feature_set)
    encoder = label_encoder or make_label_encoder()

    y = None
    if TARGET_COLUMN in df.columns:
        y = encoder.transform(df[TARGET_COLUMN])

    return X, y, categorical_columns, encoder


if __name__ == "__main__":
    train_df, test_df, _sample = load_raw()
    X_train, y_train, categorical_columns, encoder = build_features(train_df)
    X_test, _y_test, _test_categorical_columns, _ = build_features(test_df, label_encoder=encoder)

    print(f"train shape: {X_train.shape}")
    print(f"test shape: {X_test.shape}")
    print(f"categorical columns: {categorical_columns}")
    print("dtypes:")
    print(X_train.dtypes)
    print("label counts:")
    print(pd.Series(encoder.inverse_transform(y_train)).value_counts())
