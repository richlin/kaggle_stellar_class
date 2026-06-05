# Local 0.971 Score Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runnable experiment scaffolding for the remaining plausible routes to honest local OOF above `0.971`.

**Architecture:** Keep the existing competition-only incumbent intact. Add separate scripts for external-labelled spatial reference features, optional TabPFN meta-stacking, and external-catalog feature ingestion, each with fail-closed guards and JSON ledgers. Score acceptance remains competition-train OOF only.

**Tech Stack:** Python, pandas, numpy, scikit-learn, LightGBM, existing `src.spatial` and `src.validation`; TabPFN is optional and must fail cleanly if not installed.

---

### Task 52: External-Labelled Spatial Reference Append

**Files:**
- Create: `src/external_spatial.py`
- Create: `tests/test_external_spatial.py`
- Create: `scripts/47_external_spatial_append.py`

- [ ] Write tests proving competition validation labels do not affect their own OOF spatial features, external labels do affect those features, and source weights are applied only to external rows.
- [ ] Implement fold-aware spatial feature construction where each validation fold uses `competition-train-fold + audited original rows` as labelled neighbours.
- [ ] Implement a weighted append sweep over original-row weights, with competition-only OOF scoring and submission generation only if the honest OOF gate passes.

### Task 53: Optional TabPFN Meta-Stacker

**Files:**
- Create: `tests/test_tabpfn_meta_stacker.py`
- Create: `scripts/48_tabpfn_meta_stacker.py`

- [ ] Write tests for probability-to-logit feature construction and missing-TabPFN fail-closed behaviour.
- [ ] Implement a script that uses `from tabpfn import TabPFNClassifier` only when installed; otherwise write a BLOCKED experiment JSON and exit cleanly.
- [ ] Use nested competition-only folds and probability-cache logits so no validation row enters its own meta-training data.

### Task 54: External Catalog Feature Ingestion

**Files:**
- Create: `src/external_catalog.py`
- Create: `tests/test_external_catalog.py`
- Create: `scripts/49_external_catalog_features.py`

- [ ] Write tests for id-based feature joins, sky-nearest joins, numeric-only feature selection, missing indicators, and train-median imputation.
- [ ] Implement a reusable external catalog feature builder that refuses label-like columns and prefixes all joined columns with `ext_`.
- [ ] Implement a guarded LightGBM experiment that adds external numeric features to the incumbent spatial feature matrix and accepts only if OOF beats the current best honest OOF.

### Verification

- [ ] `uv run pytest tests/test_external_spatial.py tests/test_tabpfn_meta_stacker.py tests/test_external_catalog.py -q`
- [ ] `uv run pytest -q`
- [ ] `uv run ruff check .`
- [ ] `git diff --check`
