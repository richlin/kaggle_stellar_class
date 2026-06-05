"""Tests for optional TabPFN meta-stacker helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_tabpfn_script():
    spec = importlib.util.spec_from_file_location(
        "tabpfn_meta_stacker", Path("scripts/48_tabpfn_meta_stacker.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probabilities_to_logits_are_finite_and_shape_preserving() -> None:
    stacker = _load_tabpfn_script()
    probs = np.array([[0.0, 0.5, 1.0], [0.2, 0.3, 0.5]])

    logits = stacker.probabilities_to_logits(probs)

    assert logits.shape == probs.shape
    assert np.isfinite(logits).all()
    assert logits[0, 2] > logits[0, 1] > logits[0, 0]


def test_build_meta_features_concatenates_named_probability_blocks() -> None:
    stacker = _load_tabpfn_script()
    blocks = {
        "a": np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]),
        "b": np.array([[0.1, 0.8, 0.1], [0.4, 0.4, 0.2]]),
    }

    features, names = stacker.build_meta_features(blocks)

    assert features.shape == (2, 6)
    assert names == [
        "a_logit_c0",
        "a_logit_c1",
        "a_logit_c2",
        "b_logit_c0",
        "b_logit_c1",
        "b_logit_c2",
    ]


def test_load_tabpfn_classifier_reports_missing_dependency() -> None:
    stacker = _load_tabpfn_script()

    try:
        classifier = stacker.load_tabpfn_classifier()
    except RuntimeError as exc:
        assert "tabpfn" in str(exc).lower()
    else:
        assert classifier.__name__ == "TabPFNClassifier"
