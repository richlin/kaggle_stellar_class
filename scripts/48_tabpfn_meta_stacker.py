"""Optional TabPFN meta-stacker over cached model probabilities.

This is intentionally optional: if ``tabpfn`` is not installed, the script writes
a BLOCKED experiment record and exits cleanly. When installed, it trains a
competition-only meta-learner on logit-transformed OOF probability caches.
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
from sklearn.model_selection import StratifiedKFold

from src.data import load_raw, make_label_encoder
from src.validate import validate_submission
from src.validation import per_class_recall, search_class_multipliers, write_json

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
CV_N_SPLITS = 5
CV_SEED = 42
INCUMBENT_OOF = 0.969211
EXPERIMENT = PROJECT_ROOT / "experiments" / "48_tabpfn_meta_stacker.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "48_tabpfn_meta_stacker.csv"

PROBABILITY_BLOCKS = {
    "spatial_lgbm5": (
        PROJECT_ROOT / "experiments" / "32_spatial_5seed_lgbm_oof_probabilities.npy",
        PROJECT_ROOT / "experiments" / "32_spatial_5seed_lgbm_test_probabilities.npy",
    ),
    "spatial_xgb": (
        PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy",
        PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy",
    ),
    "catboost_spatial": (
        PROJECT_ROOT / "experiments" / "40_catboost_spatial_oof_probabilities.npy",
        PROJECT_ROOT / "experiments" / "40_catboost_spatial_test_probabilities.npy",
    ),
    "galactic_lgbm": (
        PROJECT_ROOT / "experiments" / "45_galactic_oof_probabilities.npy",
        PROJECT_ROOT / "experiments" / "45_galactic_test_probabilities.npy",
    ),
}


def probabilities_to_logits(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to finite logits."""
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def build_meta_features(blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Concatenate logit-transformed probability blocks."""
    matrices: list[np.ndarray] = []
    names: list[str] = []
    n_rows: int | None = None
    for name, probabilities in blocks.items():
        if probabilities.ndim != 2:
            raise ValueError(f"{name} probabilities must be 2D")
        if n_rows is None:
            n_rows = probabilities.shape[0]
        elif probabilities.shape[0] != n_rows:
            raise ValueError("all probability blocks must have the same row count")
        matrices.append(probabilities_to_logits(probabilities))
        names.extend([f"{name}_logit_c{idx}" for idx in range(probabilities.shape[1])])
    return np.hstack(matrices), names


def load_tabpfn_classifier():
    """Return TabPFNClassifier or raise a clear optional-dependency error."""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError(
            "tabpfn is not installed. Install it in the active environment before "
            "running scripts/48_tabpfn_meta_stacker.py."
        ) from exc
    return TabPFNClassifier


def _available_probability_blocks() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof_blocks: dict[str, np.ndarray] = {}
    test_blocks: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name, (oof_path, test_path) in PROBABILITY_BLOCKS.items():
        if not oof_path.exists() or not test_path.exists():
            missing.append(name)
            continue
        oof_blocks[name] = np.load(oof_path)
        test_blocks[name] = np.load(test_path)
    if len(oof_blocks) < 2:
        raise FileNotFoundError(f"need at least two probability blocks; missing={missing}")
    return oof_blocks, test_blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    try:
        TabPFNClassifier = load_tabpfn_classifier()
    except RuntimeError as exc:
        record = {
            "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "gate": "BLOCKED",
            "reason": str(exc),
            "install_hint": 'uv pip install "tabpfn"',
        }
        write_json(EXPERIMENT, record)
        print(f"BLOCKED: {exc}")
        print(f"wrote {EXPERIMENT}")
        return 0

    train, _test, sample = load_raw()
    encoder = make_label_encoder()
    y = encoder.transform(train["class"])
    oof_blocks, test_blocks = _available_probability_blocks()
    X_meta, feature_names = build_meta_features(oof_blocks)
    X_test_meta, _ = build_meta_features(test_blocks)

    oof = np.zeros((len(X_meta), 3))
    skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=CV_SEED)
    for fold, (tr, va) in enumerate(skf.split(X_meta, y), 1):
        print(f"tabpfn fold {fold}/{CV_N_SPLITS}")
        model = TabPFNClassifier()
        model.fit(X_meta[tr], y[tr])
        oof[va] = model.predict_proba(X_meta[va])

    model = TabPFNClassifier()
    model.fit(X_meta, y)
    test_prob = model.predict_proba(X_test_meta)

    mult, tuned_score = search_class_multipliers(y, oof)
    pred = (oof * mult).argmax(1)
    recalls = per_class_recall(y, pred, CLASS_LABELS)
    gate = "PASSED" if tuned_score > INCUMBENT_OOF else "FAILED"

    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "available_blocks": sorted(oof_blocks),
        "feature_names": feature_names,
        "tuned_oof": tuned_score,
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": mult.tolist(),
        "per_class_recall": recalls,
    }

    if gate == "FAILED":
        print(f"FAILED gate: {tuned_score:.6f} <= {INCUMBENT_OOF:.6f}")
        write_json(EXPERIMENT, record)
        return 0

    predicted = (test_prob * mult).argmax(1)
    submission = pd.DataFrame({"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(predicted)})
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)
    record["submission_path"] = str(SUBMISSION)
    write_json(EXPERIMENT, record)
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
