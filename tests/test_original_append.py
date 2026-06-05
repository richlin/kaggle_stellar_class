"""Tests for original-dataset append audit and training guardrails."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_original() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "objid": [101, 102, 103],
            "alpha": [10.0, 20.0, 30.0],
            "delta": [-1.0, 2.0, 3.0],
            "u": [20.0, 22.0, 19.0],
            "g": [19.0, 21.0, 18.0],
            "r": [18.0, 20.0, 17.0],
            "i": [17.5, 19.5, 16.5],
            "z": [17.0, 19.0, 16.0],
            "redshift": [0.1, 0.2, 0.0],
            "class": ["galaxy", "quasar", "stellar"],
        }
    )


def test_formula_match_detects_existing_categorical_mismatch() -> None:
    audit = _load_script("scripts/43_original_append_audit.py", "original_append_audit")
    df = _minimal_original()
    derived = audit.derive_categoricals(df)
    derived.loc[0, "spectral_type"] = "O/B"
    derived.loc[1, "galaxy_population"] = "Red_Sequence"

    result = audit.check_formula_match(derived, "fixture")

    assert result["spectral_type_mismatches"] == 1
    assert result["galaxy_population_mismatches"] == 1
    assert result["spectral_type_match_rate"] < 1.0
    assert result["galaxy_population_match_rate"] < 1.0


def test_feature_duplicate_check_detects_exact_competition_overlap() -> None:
    audit = _load_script("scripts/43_original_append_audit.py", "original_append_audit")
    original = _minimal_original()
    competition = pd.DataFrame(
        {
            "id": [999, 1000],
            "alpha": [20.0, 300.0],
            "delta": [2.0, 30.0],
            "u": [22.0, 13.0],
            "g": [21.0, 12.0],
            "r": [20.0, 11.0],
            "i": [19.5, 10.5],
            "z": [19.0, 10.0],
            "redshift": [0.2, 0.3],
        }
    )

    result = audit.check_feature_duplicates(original, competition, audit.COMPETITION_COLS)

    assert result["exact_feature_overlap"] == 1
    assert result["rounded_6dp_feature_overlap"] == 1


def test_load_original_requires_a_class_column(tmp_path: Path) -> None:
    audit = _load_script("scripts/43_original_append_audit.py", "original_append_audit")
    path = tmp_path / "no_class.csv"
    _minimal_original().drop(columns=["class"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="class column"):
        audit.load_original(str(path))


def test_training_requires_same_original_path_as_passed_audit(tmp_path: Path) -> None:
    train = _load_script("scripts/44_original_append_train.py", "original_append_train")
    audited = tmp_path / "audited.csv"
    different = tmp_path / "different.csv"
    audited.write_text("id,class\n1,GALAXY\n")
    different.write_text("id,class\n2,STAR\n")
    audit_record = {"verdict": "PASS", "original_path": str(audited)}

    train.verify_audit_matches_original(audit_record, str(audited))

    with pytest.raises(ValueError, match="does not match audit original_path"):
        train.verify_audit_matches_original(audit_record, str(different))
