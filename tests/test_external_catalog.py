"""Tests for external catalog feature ingestion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.external_catalog import (
    build_external_catalog_features,
    nearest_sky_catalog_join,
)


def test_build_external_catalog_features_aligns_by_id_and_imputes_from_train() -> None:
    train = pd.DataFrame({"id": [1, 2], "alpha": [0.0, 1.0], "class": ["A", "B"]})
    test = pd.DataFrame({"id": [3], "alpha": [2.0]})
    catalog = pd.DataFrame(
        {
            "id": [1, 3],
            "proper_motion": [10.0, 30.0],
            "morphology": [0.4, 0.8],
            "class": ["LEAK", "LEAK"],
        }
    )

    train_features, test_features, names = build_external_catalog_features(train, test, catalog, join="id")

    assert names == ["ext_morphology", "ext_proper_motion"]
    assert train_features["ext_proper_motion"].tolist() == [10.0, 10.0]
    assert train_features["ext_proper_motion_missing"].tolist() == [0.0, 1.0]
    assert test_features["ext_proper_motion"].tolist() == [30.0]
    assert "ext_class" not in train_features.columns


def test_build_external_catalog_features_rejects_label_like_feature_columns() -> None:
    train = pd.DataFrame({"id": [1]})
    test = pd.DataFrame({"id": [2]})
    catalog = pd.DataFrame({"id": [1], "target_label": [7]})

    with pytest.raises(ValueError, match="label-like"):
        build_external_catalog_features(train, test, catalog, join="id")


def test_nearest_sky_catalog_join_matches_within_arcsec_limit() -> None:
    query = pd.DataFrame({"alpha": [0.0, 10.0], "delta": [0.0, 0.0]})
    catalog = pd.DataFrame(
        {
            "alpha": [0.0001, 40.0],
            "delta": [0.0, 0.0],
            "pm": [5.0, 9.0],
        }
    )

    joined = nearest_sky_catalog_join(query, catalog, max_arcsec=1.0)

    assert joined.loc[0, "pm"] == 5.0
    assert np.isnan(joined.loc[1, "pm"])
