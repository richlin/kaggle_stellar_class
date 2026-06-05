"""Galactic coordinate features derived from equatorial RA/Dec.

Galactic latitude b encodes the object's angular distance from the Milky Way
disk. Analysis of the competition data shows that objects at |b|<10° are 98.9%
STAR, while |b|>30° regions are 66.7% GALAXY — a much stronger individual-
feature signal than raw RA/Dec for the GALAXY/STAR confusion boundary.

Unlike spatial k-NN, galactic coordinates provide a smooth, continuous signal
the model can learn with simple axis-aligned splits.
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


def add_galactic_features(
    df: pd.DataFrame,
    ra_col: str = "alpha",
    dec_col: str = "delta",
) -> pd.DataFrame:
    """Return a copy of df with galactic l, b, and derived features appended."""
    coords = SkyCoord(
        ra=df[ra_col].values * u.degree,
        dec=df[dec_col].values * u.degree,
        frame="icrs",
    )
    gal = coords.galactic
    l_deg = gal.l.degree  # 0..360
    b_deg = gal.b.degree  # -90..90

    df = df.copy()
    df["gal_b"] = b_deg
    df["gal_abs_b"] = np.abs(b_deg)
    df["gal_sin_b"] = np.sin(np.deg2rad(b_deg))
    df["gal_cos_b"] = np.cos(np.deg2rad(b_deg))
    df["gal_l"] = l_deg
    df["gal_sin_l"] = np.sin(np.deg2rad(l_deg))
    df["gal_cos_l"] = np.cos(np.deg2rad(l_deg))
    return df


GALACTIC_FEATURE_NAMES = [
    "gal_b",
    "gal_abs_b",
    "gal_sin_b",
    "gal_cos_b",
    "gal_l",
    "gal_sin_l",
    "gal_cos_l",
]
