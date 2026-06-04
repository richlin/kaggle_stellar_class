"""Tests for the shared feature-building layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import build_features

EXPECTED_ENGINEERED_COLUMNS = {
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "u_r",
    "u_i",
    "u_z",
    "g_i",
    "g_z",
    "r_z",
    "mag_mean",
    "mag_std",
    "mag_min",
    "mag_max",
    "mag_range",
    "alpha_sin",
    "alpha_cos",
    "redshift_x_u_g",
    "redshift_x_g_r",
    "redshift_x_r_i",
    "redshift_x_i_z",
    "spectral_population",
}


def _train_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "alpha": [0.0, 90.0, 180.0],
            "delta": [-10.0, 5.0, 30.0],
            "u": [20.0, 21.0, 22.0],
            "g": [19.0, 20.0, 21.0],
            "r": [18.5, 19.5, 20.5],
            "i": [18.0, 19.0, 20.0],
            "z": [17.5, 18.5, 19.5],
            "redshift": [0.05, 0.7, 1.8],
            "spectral_type": ["M", "G/K", "A/F"],
            "galaxy_population": ["Red_Sequence", "Blue_Cloud", "Red_Sequence"],
            "class": ["STAR", "GALAXY", "QSO"],
        }
    )


def _test_fixture() -> pd.DataFrame:
    return _train_fixture().drop(columns=["class"]).assign(id=[4, 5, 6])


def test_build_features_returns_expected_columns_and_categoricals() -> None:
    X, y, categorical_columns, encoder = build_features(_train_fixture())

    assert EXPECTED_ENGINEERED_COLUMNS.issubset(X.columns)
    assert "id" not in X.columns
    assert list(y) == encoder.transform(["STAR", "GALAXY", "QSO"]).tolist()
    assert set(categorical_columns) == {"spectral_type", "galaxy_population", "spectral_population"}
    for column in categorical_columns:
        assert str(X[column].dtype) == "category"


def test_train_and_test_feature_columns_match() -> None:
    X_train, _y, categorical_columns, encoder = build_features(_train_fixture())
    X_test, y_test, test_categorical_columns, _ = build_features(_test_fixture(), label_encoder=encoder)

    assert y_test is None
    assert X_test.columns.tolist() == X_train.columns.tolist()
    assert test_categorical_columns == categorical_columns


def test_feature_math_introduces_no_nan_or_infinity() -> None:
    X, _y, _categorical_columns, _encoder = build_features(_train_fixture())
    numeric = X.select_dtypes(exclude=["category"])

    assert not X.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def test_label_encoder_round_trips_all_classes() -> None:
    _X, y, _categorical_columns, encoder = build_features(_train_fixture())

    assert encoder.inverse_transform(y).tolist() == ["STAR", "GALAXY", "QSO"]
