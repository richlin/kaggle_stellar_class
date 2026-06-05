"""Autoresearch eval infra (READ-ONLY — do not mutate).

Reads scripts/_auto/config.json (the TARGET), weight-blends the named cached base-model
OOF probability arrays, tunes per-class multipliers to maximize balanced accuracy, and
prints a single metric line:  oof_balanced_accuracy: <float>
"""
# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import load_raw
from src.validation import search_class_multipliers

CONFIG = Path(__file__).parent / "config.json"
CLASS_LABELS = ["GALAXY", "QSO", "STAR"]

# name -> cached OOF probability array (all aligned, shape (577347, 3))
REGISTRY = {
    "lgbm_final": "experiments/03_final_oof_probabilities.npy",
    "xgboost": "experiments/04_xgboost_oof_probabilities.npy",
    "catboost": "experiments/04_catboost_oof_probabilities.npy",
    "lgbm_dart": "experiments/04_lgbm_dart_oof_probabilities.npy",
    "boundary_v1": "experiments/05_boundary_v1_oof_probabilities.npy",
    "tuned_xgb": "experiments/05_tuned_xgb_oof_probabilities.npy",
    "extended_seed": "experiments/09_extended_seed_average_oof_probabilities.npy",
    "unweighted_lgbm": "experiments/09_unweighted_lgbm_oof_probabilities.npy",
    "sqrt_balanced": "experiments/10_sqrt_balanced_lgbm_oof_probabilities.npy",
    "target_encoding": "experiments/10_target_encoding_oof_probabilities.npy",
    "class_weight": "experiments/13_class_weight_lgbm_oof_probabilities.npy",
}

_Y = None


def _labels() -> np.ndarray:
    global _Y
    if _Y is None:
        train, _t, _s = load_raw()
        _Y = pd.Categorical(train["class"], categories=CLASS_LABELS).codes.astype(int)
    return _Y


def main() -> int:
    weights: dict[str, float] = json.loads(CONFIG.read_text())["weights"]
    active = {k: float(v) for k, v in weights.items() if float(v) != 0.0}
    if not active:
        print("oof_balanced_accuracy: 0.0")
        return 0

    total = sum(active.values())
    blended = None
    for name, w in active.items():
        arr = np.load(PROJECT_ROOT / REGISTRY[name])
        blended = arr * (w / total) if blended is None else blended + arr * (w / total)

    y = _labels()
    _mult, score = search_class_multipliers(y, blended)
    print(f"oof_balanced_accuracy: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
