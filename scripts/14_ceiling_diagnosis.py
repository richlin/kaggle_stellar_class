"""Diagnose the balanced-accuracy ceiling: how much error is irreducible?

Three independent estimates of the Bayes floor, plus boundary characterization:
  1. Model-confidence: on the best OOF probabilities, how confident is the model
     when it is WRONG? Low-confidence errors = genuine overlap; high-confidence
     errors = label noise / systematic miss.
  2. k-NN label disagreement: a non-parametric Bayes-error estimator. In feature
     space, what is the optimal (neighbourhood-majority) classifier's implied
     per-class recall and balanced accuracy? This is model-agnostic.
  3. Exact feature collisions: rows with (rounded) identical features but
     conflicting labels — a hard lower bound on achievable error.

Reads only; writes experiments/14_ceiling_diagnosis.json.
"""
# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data import load_raw
from src.validation import balanced_accuracy, per_class_recall, write_json

RNG = np.random.default_rng(42)
CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CATEG = ["spectral_type", "galaxy_population"]
BEST_OOF = PROJECT_ROOT / "experiments" / "12_multi_blend_oof_probabilities.npy"
OUT = PROJECT_ROOT / "experiments" / "14_ceiling_diagnosis.json"

# k-NN estimator config
INDEX_N = 150_000      # rows used to build the neighbour index
QUERY_PER_CLASS = 12_000  # stratified query points per class
K = 25                 # neighbours (excluding self)


def confidence_diagnostic(y: np.ndarray, oof: np.ndarray) -> dict:
    """How confident is the model when wrong? (raw argmax, no multipliers)."""
    pred = oof.argmax(axis=1)
    correct = pred == y
    true_prob = oof[np.arange(len(y)), y]          # prob assigned to TRUE class
    pred_prob = oof.max(axis=1)
    margin = pred_prob - np.sort(oof, axis=1)[:, -2]  # top1 - top2

    err = ~correct
    return {
        "raw_argmax_balanced_accuracy": balanced_accuracy(y, pred),
        "n_errors": int(err.sum()),
        "error_rate": float(err.mean()),
        "mean_true_class_prob_on_errors": float(true_prob[err].mean()),
        "median_true_class_prob_on_errors": float(np.median(true_prob[err])),
        # errors where the model still gave the true class a real chance => overlap
        "frac_errors_trueprob_gt_0.30": float((true_prob[err] > 0.30).mean()),
        "frac_errors_trueprob_gt_0.45": float((true_prob[err] > 0.45).mean()),
        # high-confidence-wrong => label-noise / systematic candidates
        "frac_errors_predprob_gt_0.90": float((pred_prob[err] > 0.90).mean()),
        "frac_errors_predprob_gt_0.99": float((pred_prob[err] > 0.99).mean()),
        "mean_margin_correct": float(margin[correct].mean()),
        "mean_margin_error": float(margin[err].mean()),
    }


def knn_bayes_estimate(df: pd.DataFrame, y: np.ndarray) -> dict:
    """Optimal neighbourhood classifier's implied recall/balanced accuracy."""
    num = StandardScaler().fit_transform(df[NUMERIC].to_numpy(dtype=float))
    cat = pd.get_dummies(df[CATEG].astype("category"), dtype=float).to_numpy()
    feat = np.hstack([num, cat])

    idx = RNG.choice(len(feat), size=min(INDEX_N, len(feat)), replace=False)
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", n_jobs=-1)
    nn.fit(feat[idx])
    y_idx = y[idx]

    # stratified query sample so per-class recall is well estimated
    q_parts = []
    for c in range(3):
        c_rows = np.where(y == c)[0]
        q_parts.append(RNG.choice(c_rows, size=min(QUERY_PER_CLASS, len(c_rows)), replace=False))
    q = np.concatenate(q_parts)

    _, neigh = nn.kneighbors(feat[q])
    neigh = neigh[:, 1:]                    # drop nearest (likely self / dup)
    neigh_labels = y_idx[neigh]             # (n_query, K)

    # Bayes-optimal prediction = neighbourhood majority
    counts = np.stack([(neigh_labels == c).sum(axis=1) for c in range(3)], axis=1)
    bayes_pred = counts.argmax(axis=1)
    yq = y[q]

    recalls = per_class_recall(yq, bayes_pred, CLASS_LABELS)
    # local irreducible error = fraction of neighbours NOT in the majority class
    local_err = 1.0 - counts.max(axis=1) / K
    return {
        "k": K,
        "index_size": int(len(idx)),
        "query_size": int(len(q)),
        "bayes_optimal_balanced_accuracy": balanced_accuracy(yq, bayes_pred),
        "bayes_optimal_per_class_recall": recalls,
        "mean_local_irreducible_error": float(local_err.mean()),
        "mean_local_irreducible_error_by_true_class": {
            CLASS_LABELS[c]: float(local_err[yq == c].mean()) for c in range(3)
        },
    }


def collision_estimate(df: pd.DataFrame, y: np.ndarray) -> dict:
    """Hard lower bound: rounded-identical feature rows with conflicting labels."""
    key = pd.DataFrame({
        "redshift": (df["redshift"] / 0.02).round().astype(int),
        "u_g": ((df["u"] - df["g"]) / 0.1).round().astype(int),
        "g_r": ((df["g"] - df["r"]) / 0.1).round().astype(int),
        "r_i": ((df["r"] - df["i"]) / 0.1).round().astype(int),
        "i_z": ((df["i"] - df["z"]) / 0.1).round().astype(int),
        "r": (df["r"] / 0.25).round().astype(int),
        "spectral_type": df["spectral_type"].astype(str),
        "galaxy_population": df["galaxy_population"].astype(str),
        "_y": y,
    })
    grp = key.groupby(["redshift", "u_g", "g_r", "r_i", "i_z", "r",
                       "spectral_type", "galaxy_population"])
    sizes = grp["_y"].size()
    majority = grp["_y"].agg(lambda s: s.value_counts().iloc[0])
    # min errors = total - sum of per-cell majority counts
    min_errors = int((sizes - majority).sum())
    multi = sizes[sizes > 1]
    mixed = (majority[multi.index] < multi)
    return {
        "tolerance": "redshift 0.02, colors 0.1 mag, r 0.25 mag",
        "n_cells": int(len(sizes)),
        "n_cells_multi_row": int(len(multi)),
        "n_cells_mixed_label": int(mixed.sum()),
        "min_errors_lower_bound": min_errors,
        "min_error_rate_lower_bound": float(min_errors / len(y)),
    }


def boundary_characterization(df: pd.DataFrame, y: np.ndarray, oof: np.ndarray) -> dict:
    """Confirm the GALAXY<->STAR leak is the low-redshift overlap."""
    pred = oof.argmax(axis=1)
    g, s = CLASS_LABELS.index("GALAXY"), CLASS_LABELS.index("STAR")
    g_to_s = (y == g) & (pred == s)
    s_to_g = (y == s) & (pred == g)
    z = df["redshift"].to_numpy()
    return {
        "galaxy_to_star_errors": int(g_to_s.sum()),
        "frac_g2s_redshift_lt_0.15": float((z[g_to_s] < 0.15).mean()),
        "median_redshift_g2s": float(np.median(z[g_to_s])),
        "star_to_galaxy_errors": int(s_to_g.sum()),
        "frac_s2g_redshift_lt_0.15": float((z[s_to_g] < 0.15).mean()),
        "median_redshift_true_galaxy": float(np.median(z[y == g])),
        "median_redshift_true_star": float(np.median(z[y == s])),
    }


def main() -> int:
    train, _test, _sample = load_raw()
    y = pd.Categorical(train["class"], categories=CLASS_LABELS).codes.astype(int)
    oof = np.load(BEST_OOF)
    assert len(oof) == len(y), "OOF/label length mismatch"

    print("1/4 confidence diagnostic ...")
    conf = confidence_diagnostic(y, oof)
    print("2/4 k-NN Bayes estimate ...")
    knn = knn_bayes_estimate(train, y)
    print("3/4 collision lower bound ...")
    coll = collision_estimate(train, y)
    print("4/4 boundary characterization ...")
    bnd = boundary_characterization(train, y, oof)

    current_best_oof = 0.966282
    bayes_ba = knn["bayes_optimal_balanced_accuracy"]
    record = {
        "current_best_local_oof": current_best_oof,
        "target": 0.97,
        "gap_to_target": round(0.97 - current_best_oof, 6),
        "bayes_optimal_balanced_accuracy_knn": bayes_ba,
        "headroom_to_bayes_ceiling": round(bayes_ba - current_best_oof, 6),
        "confidence_diagnostic": conf,
        "knn_bayes_estimate": knn,
        "collision_lower_bound": coll,
        "boundary": bnd,
    }
    write_json(OUT, record)

    print("\n================ CEILING DIAGNOSIS ================")
    print(f"current best local OOF      : {current_best_oof:.6f}")
    print(f"target                      : 0.970000  (gap {0.97 - current_best_oof:+.6f})")
    print(f"k-NN Bayes-optimal bal. acc : {bayes_ba:.6f}")
    print(f"  -> headroom to ceiling    : {bayes_ba - current_best_oof:+.6f}")
    print(f"  per-class Bayes recall    : {knn['bayes_optimal_per_class_recall']}")
    print(f"hard collision min err-rate : {coll['min_error_rate_lower_bound']:.6f} "
          f"(=> bal-acc upper bound ~ {1 - coll['min_error_rate_lower_bound']:.4f})")
    print(f"mean true-class prob on errs: {conf['mean_true_class_prob_on_errors']:.3f}")
    print(f"frac errors w/ pred-prob>.99: {conf['frac_errors_predprob_gt_0.99']:.3f} "
          f"(high-confidence-wrong = label-noise candidates)")
    print(f"GALAXY->STAR errs <z0.15    : {bnd['frac_g2s_redshift_lt_0.15']:.3f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
