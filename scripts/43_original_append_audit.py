"""Task 47: audit the original SDSS dataset for safe append to competition training data.

Recreates spectral_type and galaxy_population from the competition formulae, aligns
the schema to the competition train DataFrame, checks for duplicate ids/features
against competition train+test, checks class distribution, and checks feature shift.

Writes experiments/43_original_append_audit.json with a PASS/FAIL verdict and
detailed diagnostics.

Usage:
  uv run python scripts/43_original_append_audit.py --original /path/to/original.csv

The original file must have columns: objid (or id), alpha, delta, u, g, r, i, z,
redshift, and a class column (STAR/GALAXY/QSO or equivalent).
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import load_raw
from src.validation import write_json

COMPETITION_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CLASS_LABELS = {"GALAXY", "QSO", "STAR"}


def derive_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive spectral_type and galaxy_population from photometry."""
    df = df.copy()
    df["spectral_type"] = pd.cut(
        df["r"] - df["g"],
        bins=[-np.inf, -1, -0.5, 0, np.inf],
        labels=["M", "G/K", "A/F", "O/B"],
    )
    df["galaxy_population"] = pd.cut(
        df["u"] - df["r"],
        bins=[-np.inf, 2.2, np.inf],
        labels=["Blue_Cloud", "Red_Sequence"],
    )
    return df


def check_formula_match(df: pd.DataFrame, source: str) -> dict:
    """Verify derived categoricals match existing ones if present."""
    result = {"source": source}
    derived = derive_categoricals(df)
    for col in ["spectral_type", "galaxy_population"]:
        if col not in df.columns:
            result[f"{col}_match"] = "not_present"
            continue
        matches = df[col].astype(str) == derived[col].astype(str)
        match_rate = matches.mean()
        result[f"{col}_match_rate"] = float(match_rate)
        result[f"{col}_mismatches"] = int((~matches).sum())
    return result


def load_original(path: str) -> tuple[pd.DataFrame, str]:
    """Load original dataset and normalise class labels."""
    df = pd.read_csv(path)
    # Try common class column names
    for col in ["class", "Class", "CLASS", "label", "Label"]:
        if col in df.columns:
            # Normalise class values
            df = df.rename(columns={col: "class"})
            break
    if "class" not in df.columns:
        raise ValueError(f"Could not find class column in {path}. Columns: {list(df.columns)}")

    # Normalise class labels to GALAXY/QSO/STAR
    cls_map = {}
    for v in df["class"].unique():
        vstr = str(v).upper().strip()
        if vstr in CLASS_LABELS:
            cls_map[v] = vstr
        elif "GAL" in vstr:
            cls_map[v] = "GALAXY"
        elif "QSO" in vstr or "QUASAR" in vstr:
            cls_map[v] = "QSO"
        elif "STAR" in vstr or "STELLAR" in vstr:
            cls_map[v] = "STAR"
    df["class"] = df["class"].map(cls_map)

    unknown = df["class"].isna().sum()
    if unknown > 0:
        print(f"  WARNING: {unknown} rows have unrecognised class labels; dropping them.")
        df = df.dropna(subset=["class"])

    # Normalise id column
    for col in ["objid", "obj_id", "specobjid", "spec_obj_id", "id"]:
        if col in df.columns:
            df = df.rename(columns={col: "id"})
            break

    return df, "loaded"


def check_feature_duplicates(orig: pd.DataFrame, comp: pd.DataFrame, cols: list[str]) -> dict:
    """Check whether original rows duplicate competition feature rows."""
    exact_orig = set(map(tuple, orig[cols].to_numpy()))
    exact_comp = set(map(tuple, comp[cols].to_numpy()))

    rounded_orig = set(map(tuple, orig[cols].round(6).to_numpy()))
    rounded_comp = set(map(tuple, comp[cols].round(6).to_numpy()))

    return {
        "exact_feature_overlap": len(exact_orig.intersection(exact_comp)),
        "rounded_6dp_feature_overlap": len(rounded_orig.intersection(rounded_comp)),
    }


def check_feature_shift(orig: pd.DataFrame, comp: pd.DataFrame, cols: list[str]) -> dict:
    """Compare feature distributions between original and competition data."""
    results = {}
    for col in cols:
        if col not in orig.columns or col not in comp.columns:
            continue
        orig_vals = orig[col].dropna()
        comp_vals = comp[col].dropna()
        # Kolmogorov-Smirnov approximation: compare percentiles
        p_orig = np.percentile(orig_vals, [5, 25, 50, 75, 95])
        p_comp = np.percentile(comp_vals, [5, 25, 50, 75, 95])
        max_pct_diff = float(np.max(np.abs(p_orig - p_comp) / (np.abs(p_comp) + 1e-9)))
        results[col] = {
            "orig_mean": float(orig_vals.mean()),
            "comp_mean": float(comp_vals.mean()),
            "orig_std": float(orig_vals.std()),
            "comp_std": float(comp_vals.std()),
            "max_percentile_rel_diff": max_pct_diff,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to original labeled dataset CSV")
    args = parser.parse_args()

    print(f"Loading original dataset from {args.original} ...")
    orig, status = load_original(args.original)
    print(f"  {len(orig)} rows, {len(orig.columns)} cols")
    print(f"  Class distribution: {orig['class'].value_counts().to_dict()}")

    print("Loading competition train+test ...")
    comp_train, comp_test, _ = load_raw()
    comp_all = pd.concat([
        comp_train[["id"] + COMPETITION_COLS],
        comp_test[["id"] + COMPETITION_COLS],
    ], ignore_index=True)
    print(f"  Competition: {len(comp_train)} train + {len(comp_test)} test = {len(comp_all)} total")

    results: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "original_path": str(args.original),
        "original_rows": len(orig),
        "original_class_distribution": orig["class"].value_counts().to_dict(),
        "competition_rows": len(comp_all),
    }

    # 1. Check duplicate ids
    print("Checking duplicate ids ...")
    if "id" in orig.columns:
        overlap_ids = set(orig["id"]).intersection(set(comp_all["id"]))
        results["id_overlap"] = len(overlap_ids)
        if overlap_ids:
            print(f"  FAIL: {len(overlap_ids)} duplicate ids found with competition data!")
        else:
            print("  PASS: no id overlap")
    else:
        results["id_overlap"] = "no_id_column"
        print("  INFO: no id column in original — cannot check id overlap")

    # 2. Check missing required columns
    missing_cols = [c for c in COMPETITION_COLS if c not in orig.columns]
    results["missing_required_cols"] = missing_cols
    if missing_cols:
        print(f"  FAIL: missing required columns: {missing_cols}")
        results["verdict"] = "FAIL"
        results["fail_reason"] = f"missing columns: {missing_cols}"
        write_json(PROJECT_ROOT / "experiments" / "43_original_append_audit.json", results)
        return 1
    print("  PASS: all required columns present")

    # 3. Derive categoricals and verify formula
    print("Deriving and checking categoricals ...")
    formula_match = check_formula_match(orig, "original")
    results["categorical_formula_match"] = formula_match
    orig = derive_categoricals(orig)

    # 4. Check class distribution shift
    orig_dist = orig["class"].value_counts(normalize=True).to_dict()
    comp_dist = comp_train["class"].value_counts(normalize=True).to_dict() if "class" in comp_train.columns else {}
    results["original_class_fractions"] = {k: float(v) for k, v in orig_dist.items()}
    results["competition_train_class_fractions"] = {k: float(v) for k, v in comp_dist.items()}

    if comp_dist:
        max_class_diff = max(abs(orig_dist.get(c, 0) - comp_dist.get(c, 0)) for c in CLASS_LABELS)
        results["max_class_distribution_diff"] = float(max_class_diff)
        print(f"  Max class distribution diff: {max_class_diff:.4f}")
    else:
        results["max_class_distribution_diff"] = "no_comp_labels"

    # 5. Feature shift check
    print("Checking feature shift ...")
    shift = check_feature_shift(orig, comp_all, COMPETITION_COLS)
    results["feature_shift"] = shift
    max_shift = max(v["max_percentile_rel_diff"] for v in shift.values() if isinstance(v, dict))
    results["max_feature_shift"] = float(max_shift)
    print(f"  Max relative feature shift: {max_shift:.4f}")

    # 6. Coordinate overlap check (approximate — check for exact matches)
    print("Checking coordinate duplicates ...")
    if "alpha" in orig.columns and "delta" in orig.columns:
        orig_coords = set(zip(orig["alpha"].round(4), orig["delta"].round(4)))
        comp_coords = set(zip(comp_all["alpha"].round(4), comp_all["delta"].round(4)))
        coord_overlap = len(orig_coords.intersection(comp_coords))
        results["coordinate_overlap_approx"] = coord_overlap
        print(f"  Approximate coordinate overlap (4dp): {coord_overlap} rows")
    else:
        results["coordinate_overlap_approx"] = "no_coords"

    print("Checking full feature duplicates ...")
    feature_duplicates = check_feature_duplicates(orig, comp_all, COMPETITION_COLS)
    results["feature_duplicate_check"] = feature_duplicates
    print(
        "  Exact feature overlap: "
        f"{feature_duplicates['exact_feature_overlap']}; "
        "rounded 6dp overlap: "
        f"{feature_duplicates['rounded_6dp_feature_overlap']}"
    )

    # 7. Final verdict
    fail_conditions = []
    if results.get("id_overlap", 0) not in (0, "no_id_column"):
        fail_conditions.append(f"id_overlap={results['id_overlap']}")
    if missing_cols:
        fail_conditions.append(f"missing_cols={missing_cols}")
    for col in ["spectral_type", "galaxy_population"]:
        mismatches = formula_match.get(f"{col}_mismatches")
        if mismatches:
            fail_conditions.append(f"{col}_formula_mismatches={mismatches}")
    if feature_duplicates["exact_feature_overlap"] > 0:
        fail_conditions.append(f"exact_feature_overlap={feature_duplicates['exact_feature_overlap']}")
    if feature_duplicates["rounded_6dp_feature_overlap"] > 0:
        fail_conditions.append(
            f"rounded_6dp_feature_overlap={feature_duplicates['rounded_6dp_feature_overlap']}"
        )
    if max_shift > 0.5:
        fail_conditions.append(f"large_feature_shift={max_shift:.3f}")

    if fail_conditions:
        results["verdict"] = "FAIL"
        results["fail_reasons"] = fail_conditions
        print(f"\nVERDICT: FAIL — {fail_conditions}")
    else:
        results["verdict"] = "PASS"
        results["clean_append_rows"] = len(orig)
        print(f"\nVERDICT: PASS — {len(orig)} rows ready to append")

    write_json(PROJECT_ROOT / "experiments" / "43_original_append_audit.json", results)
    print("wrote experiments/43_original_append_audit.json")
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
