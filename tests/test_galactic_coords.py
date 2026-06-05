"""Tests for galactic coordinate feature construction (src/galactic.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.galactic import GALACTIC_FEATURE_NAMES, add_galactic_features


def _make_df(alpha, delta):
    return pd.DataFrame({"alpha": alpha, "delta": delta})


def test_galactic_centre_has_low_b():
    """Galactic centre (l≈0, b≈0) is near (RA=266.4, Dec=-28.9)."""
    out = add_galactic_features(_make_df([266.4], [-28.9]))
    assert abs(out["gal_b"].iloc[0]) < 2.0


def test_galactic_pole_has_high_b():
    """North Galactic Pole (RA≈192.86, Dec≈27.13) has |b| ≈ 90°."""
    out = add_galactic_features(_make_df([192.86], [27.13]))
    assert out["gal_abs_b"].iloc[0] > 85.0


def test_galactic_features_are_finite_and_complete():
    """All galactic feature columns exist and are finite for random sky positions."""
    rng = np.random.default_rng(42)
    n = 200
    df = _make_df(rng.uniform(0, 360, n), rng.uniform(-80, 80, n))
    out = add_galactic_features(df)
    for col in GALACTIC_FEATURE_NAMES:
        assert col in out.columns, f"missing column: {col}"
        assert np.isfinite(out[col]).all(), f"{col} has non-finite values"


def test_galactic_abs_b_is_nonneg():
    """gal_abs_b must be non-negative for all inputs."""
    rng = np.random.default_rng(7)
    df = _make_df(rng.uniform(0, 360, 100), rng.uniform(-90, 90, 100))
    out = add_galactic_features(df)
    assert (out["gal_abs_b"] >= 0).all()


def test_galactic_sin_cos_b_in_range():
    """sin/cos(b) must lie in [-1, 1]."""
    rng = np.random.default_rng(13)
    df = _make_df(rng.uniform(0, 360, 300), rng.uniform(-90, 90, 300))
    out = add_galactic_features(df)
    assert (out["gal_sin_b"].between(-1, 1)).all()
    assert (out["gal_cos_b"].between(-1, 1)).all()
