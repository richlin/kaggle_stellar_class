"""External catalog feature ingestion for guarded score-push experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.spatial import radec_to_xyz

LABEL_LIKE_TOKENS = ("class", "label", "target", "y_true", "truth")
COMPETITION_COLUMNS = {
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
}


def _reject_label_like_columns(columns: list[str]) -> None:
    bad = [
        column
        for column in columns
        if any(token in column.lower() for token in LABEL_LIKE_TOKENS)
    ]
    if bad:
        raise ValueError(f"external catalog contains label-like feature columns: {bad}")


def _external_numeric_columns(catalog: pd.DataFrame, join: str) -> list[str]:
    join_columns = {"id"} if join == "id" else {"alpha", "delta"}
    candidates = [
        column
        for column in catalog.columns
        if column not in join_columns and column not in COMPETITION_COLUMNS
    ]
    numeric_candidates = [
        column
        for column in candidates
        if pd.api.types.is_numeric_dtype(catalog[column])
    ]
    _reject_label_like_columns(numeric_candidates)
    return numeric_candidates


def nearest_sky_catalog_join(
    query: pd.DataFrame,
    catalog: pd.DataFrame,
    max_arcsec: float,
) -> pd.DataFrame:
    """Join nearest catalog row to query rows within an angular distance limit."""
    if max_arcsec <= 0:
        raise ValueError("max_arcsec must be positive")
    for column in ["alpha", "delta"]:
        if column not in query.columns or column not in catalog.columns:
            raise ValueError(f"sky join requires {column!r} in query and catalog")

    feature_columns = [
        column
        for column in catalog.columns
        if column not in {"alpha", "delta"}
    ]
    if not feature_columns:
        return pd.DataFrame(index=query.index)

    query_xyz = radec_to_xyz(query["alpha"].to_numpy(), query["delta"].to_numpy())
    catalog_xyz = radec_to_xyz(catalog["alpha"].to_numpy(), catalog["delta"].to_numpy())
    nn = NearestNeighbors(n_neighbors=1).fit(catalog_xyz)
    dist, idx = nn.kneighbors(query_xyz)
    chord_limit = 2.0 * np.sin(np.deg2rad(max_arcsec / 3600.0) / 2.0)

    rows = catalog.iloc[idx[:, 0]][feature_columns].reset_index(drop=True)
    rows.loc[dist[:, 0] > chord_limit, :] = np.nan
    rows.index = query.index
    return rows


def _join_catalog(
    frame: pd.DataFrame,
    catalog: pd.DataFrame,
    join: str,
    max_arcsec: float,
) -> pd.DataFrame:
    if join == "id":
        if "id" not in frame.columns or "id" not in catalog.columns:
            raise ValueError("id join requires 'id' in frame and catalog")
        return frame[["id"]].merge(catalog, on="id", how="left").drop(columns=["id"])
    if join == "sky":
        return nearest_sky_catalog_join(frame, catalog, max_arcsec=max_arcsec)
    raise ValueError(f"unknown external catalog join mode: {join}")


def build_external_catalog_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog: pd.DataFrame,
    join: str,
    max_arcsec: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build aligned external numeric feature matrices for train and test rows."""
    numeric_columns = _external_numeric_columns(catalog, join=join)
    if not numeric_columns:
        raise ValueError("external catalog has no numeric non-label feature columns")

    join_columns = ["id"] if join == "id" else ["alpha", "delta"]
    selected = catalog[join_columns + numeric_columns].copy()
    train_joined = _join_catalog(train, selected, join, max_arcsec)[numeric_columns]
    test_joined = _join_catalog(test, selected, join, max_arcsec)[numeric_columns]

    medians = train_joined.median(numeric_only=True)
    train_features = pd.DataFrame(index=train.index)
    test_features = pd.DataFrame(index=test.index)
    names: list[str] = []
    for column in sorted(numeric_columns):
        name = f"ext_{column}"
        missing_name = f"{name}_missing"
        train_features[name] = train_joined[column].fillna(medians[column]).to_numpy()
        test_features[name] = test_joined[column].fillna(medians[column]).to_numpy()
        train_features[missing_name] = train_joined[column].isna().astype(float).to_numpy()
        test_features[missing_name] = test_joined[column].isna().astype(float).to_numpy()
        names.append(name)

    return train_features, test_features, names
