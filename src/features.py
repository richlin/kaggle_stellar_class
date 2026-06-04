"""Deterministic feature builders for the stellar-class competition."""
from __future__ import annotations

import numpy as np
import pandas as pd

PHOTOMETRIC_COLUMNS = ["u", "g", "r", "i", "z"]
BASE_NUMERIC_COLUMNS = ["alpha", "delta", *PHOTOMETRIC_COLUMNS, "redshift"]
BASE_CATEGORICAL_COLUMNS = ["spectral_type", "galaxy_population"]
ENGINEERED_CATEGORICAL_COLUMNS = ["spectral_population"]
CATEGORICAL_COLUMNS = [*BASE_CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS]


def build_feature_frame(df: pd.DataFrame, feature_set: str = "baseline") -> tuple[pd.DataFrame, list[str]]:
    """Return model-ready features and categorical column names."""
    if feature_set != "baseline":
        raise ValueError(f"unknown feature_set: {feature_set}")

    X = df[BASE_NUMERIC_COLUMNS + BASE_CATEGORICAL_COLUMNS].copy()

    X["u_g"] = X["u"] - X["g"]
    X["g_r"] = X["g"] - X["r"]
    X["r_i"] = X["r"] - X["i"]
    X["i_z"] = X["i"] - X["z"]
    X["u_r"] = X["u"] - X["r"]
    X["u_i"] = X["u"] - X["i"]
    X["u_z"] = X["u"] - X["z"]
    X["g_i"] = X["g"] - X["i"]
    X["g_z"] = X["g"] - X["z"]
    X["r_z"] = X["r"] - X["z"]

    magnitudes = X[PHOTOMETRIC_COLUMNS]
    X["mag_mean"] = magnitudes.mean(axis=1)
    X["mag_std"] = magnitudes.std(axis=1)
    X["mag_min"] = magnitudes.min(axis=1)
    X["mag_max"] = magnitudes.max(axis=1)
    X["mag_range"] = X["mag_max"] - X["mag_min"]

    alpha_radians = np.deg2rad(X["alpha"])
    X["alpha_sin"] = np.sin(alpha_radians)
    X["alpha_cos"] = np.cos(alpha_radians)

    X["redshift_x_u_g"] = X["redshift"] * X["u_g"]
    X["redshift_x_g_r"] = X["redshift"] * X["g_r"]
    X["redshift_x_r_i"] = X["redshift"] * X["r_i"]
    X["redshift_x_i_z"] = X["redshift"] * X["i_z"]

    X["spectral_population"] = (
        X["spectral_type"].astype("string") + "__" + X["galaxy_population"].astype("string")
    )

    for column in CATEGORICAL_COLUMNS:
        X[column] = X[column].astype("category")

    return X, CATEGORICAL_COLUMNS.copy()
