"""Focused tests for Phase 4 tuning helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_tune_module():
    module_path = Path("scripts/03_tune.py")
    spec = importlib.util.spec_from_file_location("tune_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_drop_feature_family_removes_columns_and_categorical_members() -> None:
    tune = _load_tune_module()
    X = pd.DataFrame(
        {
            "u": [1.0],
            "g": [1.0],
            "redshift": [0.1],
            "spectral_type": pd.Series(["M"], dtype="category"),
            "galaxy_population": pd.Series(["Red_Sequence"], dtype="category"),
            "spectral_population": pd.Series(["M__Red_Sequence"], dtype="category"),
        }
    )

    X_reduced, categorical_columns = tune.drop_feature_family(
        X,
        ["spectral_type", "galaxy_population", "spectral_population"],
        "categorical_interaction",
    )

    assert "spectral_population" not in X_reduced.columns
    assert categorical_columns == ["spectral_type", "galaxy_population"]


def test_select_final_candidate_requires_repeated_score_to_beat_reference() -> None:
    tune = _load_tune_module()
    candidates = [
        {"name": "phase3_like", "repeated_mean_chosen_oof": 0.9651},
        {"name": "regularized", "repeated_mean_chosen_oof": 0.9654},
    ]

    selected = tune.select_final_candidate(candidates, reference_score=0.965102)

    assert selected["name"] == "regularized"


def test_select_final_candidate_falls_back_when_no_candidate_beats_reference() -> None:
    tune = _load_tune_module()
    candidates = [{"name": "regularized", "repeated_mean_chosen_oof": 0.9649}]

    selected = tune.select_final_candidate(candidates, reference_score=0.965102)

    assert selected["name"] == "regularized"
    assert selected["beats_reference"] is False


def test_find_cached_run_matches_kind_and_field_value() -> None:
    tune = _load_tune_module()
    rows = [
        {"kind": "candidate_screen", "name": "phase3_like", "score": 0.965},
        {"kind": "feature_ablation", "dropped_family": "colors", "score": 0.964},
    ]

    cached = tune.find_cached_run(rows, "feature_ablation", "dropped_family", "colors")

    assert cached == {"kind": "feature_ablation", "dropped_family": "colors", "score": 0.964}
