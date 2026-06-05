"""Task 29 (Phase 11): logit-space blend of spatial LGBM and XGBoost OOF probs.

Instead of w·p1 + (1-w)·p2 (arithmetic mean), blend in log-probability space
(geometric mean after renormalisation) then retune per-class multipliers.

Acceptance gate: tuned OOF > 0.969071 (best honest OOF, 16_spatial_blend).
"""
# ruff: noqa: E402
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import build_features, load_raw
from src.validate import validate_submission
from src.validation import (
    per_class_recall,
    search_class_multipliers,
    write_json,
)

CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
INCUMBENT_OOF = 0.969071  # 16_spatial_blend

LGBM_OOF = PROJECT_ROOT / "experiments" / "15_spatial_oof_probabilities.npy"
LGBM_TEST = PROJECT_ROOT / "experiments" / "15_spatial_test_probabilities.npy"
XGB_OOF = PROJECT_ROOT / "experiments" / "16_spatial_xgb_oof_probabilities.npy"
XGB_TEST = PROJECT_ROOT / "experiments" / "16_spatial_xgb_test_probabilities.npy"
EXPERIMENT = PROJECT_ROOT / "experiments" / "29_logit_blend.json"
SUBMISSION = PROJECT_ROOT / "submissions" / "29_logit_blend.csv"

EPS = 1e-8


def logit_blend(p1: np.ndarray, p2: np.ndarray, w1: float) -> np.ndarray:
    """Blend in log-prob space (geometric mean) and renormalise."""
    log_blend = w1 * np.log(np.clip(p1, EPS, 1)) + (1 - w1) * np.log(np.clip(p2, EPS, 1))
    log_blend -= log_blend.max(axis=1, keepdims=True)  # numerical stability
    exp_l = np.exp(log_blend)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def main() -> int:
    lgbm_oof = np.load(LGBM_OOF)
    lgbm_test = np.load(LGBM_TEST)
    xgb_oof = np.load(XGB_OOF)
    xgb_test = np.load(XGB_TEST)

    train, test, sample = load_raw()
    _X, y, _cat, encoder = build_features(train)

    # reference: arithmetic blend score (script 16 baseline)
    best_arith = None
    for w in np.linspace(0, 1, 21):
        blend = w * lgbm_oof + (1 - w) * xgb_oof
        mult, score = search_class_multipliers(y, blend)
        if best_arith is None or score > best_arith["score"]:
            best_arith = {"w_lgbm": float(w), "score": score, "mult": mult}
    print(f"arithmetic blend best OOF : {best_arith['score']:.6f}  (w_lgbm={best_arith['w_lgbm']:.2f})")

    # logit blend
    best_logit = None
    for w in np.linspace(0, 1, 21):
        blend = logit_blend(lgbm_oof, xgb_oof, w)
        mult, score = search_class_multipliers(y, blend)
        if best_logit is None or score > best_logit["score"]:
            best_logit = {"w_lgbm": float(w), "score": score, "mult": mult}
    print(f"logit     blend best OOF  : {best_logit['score']:.6f}  (w_lgbm={best_logit['w_lgbm']:.2f})")
    print(f"  delta vs arithmetic     : {best_logit['score'] - best_arith['score']:+.6f}")
    print(f"  delta vs incumbent      : {best_logit['score'] - INCUMBENT_OOF:+.6f}")

    gate = "PASSED" if best_logit["score"] > INCUMBENT_OOF else "FAILED"
    record: dict = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "arithmetic_blend_oof": best_arith["score"],
        "arithmetic_w_lgbm": best_arith["w_lgbm"],
        "logit_blend_oof": best_logit["score"],
        "logit_w_lgbm": best_logit["w_lgbm"],
        "incumbent_oof": INCUMBENT_OOF,
        "gate": gate,
        "multipliers": best_logit["mult"].tolist(),
    }

    if gate == "FAILED":
        print(f"\nFAILED acceptance gate ({INCUMBENT_OOF:.6f}) — not writing submission")
        write_json(EXPERIMENT, record)
        return 0

    # gate passed — write submission
    w = best_logit["w_lgbm"]
    blend_test = logit_blend(lgbm_test, xgb_test, w)
    mult = best_logit["mult"]
    pred = (blend_test * mult).argmax(1)
    recalls = per_class_recall(y, (logit_blend(lgbm_oof, xgb_oof, w) * mult).argmax(1), CLASS_LABELS)

    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(), "class": encoder.inverse_transform(pred)}
    )
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION, index=False)
    validate_submission(SUBMISSION, sample)
    record["per_class_recall"] = recalls
    record["submission_path"] = str(SUBMISSION)
    write_json(EXPERIMENT, record)
    print(f"wrote {SUBMISSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
