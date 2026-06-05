# Task 24 Transductive Spatial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a score candidate that can push the public score beyond `0.97` by adding graph-style spatial probabilities, multi-resolution cluster class rates, and a GALAXY-focused residual calibrator.

**Architecture:** Keep the current `16_spatial_blend.csv` model as the public incumbent and add a narrow Phase 9 script that consumes its cached probabilities. New reusable feature builders live in `src/transductive_spatial.py`; the candidate runner lives in `scripts/17_transductive_spatial.py`; tests cover leakage controls and submission shape.

**Tech Stack:** Python, NumPy, pandas, scikit-learn `NearestNeighbors` / `MiniBatchKMeans`, LightGBM, existing `src.validation` and `src.validate` helpers.

---

## File Structure

- Create `src/transductive_spatial.py`: weighted kNN probability features, OOF-safe cluster class-rate features, and probability meta-feature helpers.
- Create `tests/test_transductive_spatial.py`: unit tests for self-neighbour exclusion, OOF cluster leakage, meta-feature shape, and submission id order.
- Create `scripts/17_transductive_spatial.py`: cached feature generation, residual LightGBM CV, blend search, threshold tuning, experiment JSON, and submission CSV.
- Modify `tasks/todo.md`: mark Task 24 scope and acceptance gates.
- Modify `PROGRESS.md`, `DECISIONS.md`, and `experiments/leaderboard.md` after the run records a concrete candidate.

## Task 1: Tests First

**Files:**
- Create: `tests/test_transductive_spatial.py`

- [ ] Write tests that import `src.transductive_spatial` and assert:
  - graph probability features exclude a query row's own soft label when test rows are part of the reference graph;
  - OOF cluster class-rate features do not change for a validation row when that same row's own label is flipped;
  - probability meta-features return finite train/test matrices with matching columns;
  - the script helper `make_submission()` preserves sample id order.
- [ ] Run `uv run pytest tests/test_transductive_spatial.py -q` and confirm it fails because the module/script helper is missing.

## Task 2: Feature Builder

**Files:**
- Create: `src/transductive_spatial.py`

- [ ] Implement `weighted_graph_probabilities()` with inverse-distance weights, finite normalization, and optional self-reference exclusion.
- [ ] Implement `oof_cluster_class_rates()` and `test_cluster_class_rates()` using precomputed cluster ids, class priors, and smoothing.
- [ ] Implement `build_probability_meta_features()` for base probabilities, margins, entropy, and class-probability gaps.
- [ ] Run `uv run pytest tests/test_transductive_spatial.py -q` and confirm the non-script tests pass.

## Task 3: Candidate Script

**Files:**
- Create: `scripts/17_transductive_spatial.py`

- [ ] Load raw data, cached `15_spatial_*` and `16_spatial_xgb_*` probabilities, and cached spatial feature arrays.
- [ ] Build graph probability features from unit-sphere coordinates using k values `10, 25, 50, 100, 250`.
- [ ] Build multi-resolution cluster class-rate features at `512`, `2048`, and `8192` clusters with OOF-safe train rates and full-train test rates.
- [ ] Train a residual LightGBM meta-classifier on meta features only, then blend its OOF/test probabilities with the existing `16_spatial_blend` probabilities.
- [ ] Search blend weights and per-class multipliers on OOF balanced accuracy.
- [ ] Write `submissions/17_transductive_spatial.csv`, `experiments/17_transductive_spatial.json`, and cached probability arrays.

## Task 4: Verification and Documentation

**Files:**
- Modify: `PROGRESS.md`
- Modify: `DECISIONS.md`
- Modify: `experiments/leaderboard.md`

- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run python src/validate.py submissions/17_transductive_spatial.csv`.
- [ ] Record whether `17_transductive_spatial` beats the current OOF incumbent `0.9690706512708674`, the per-class recalls, and whether it is worth submitting.
