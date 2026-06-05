# Score Over 0.97 Revisit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revisit the stellar-classification score push and produce a candidate that can beat the current public incumbent `0.96970` and clear `0.97`.

**Architecture:** Stop submitting class-multiplier variants from the existing cached `19` and `25` probability files. Build one new spatial signal at a time, verify it with leakage-safe OOF where possible, then create at most two public-risk final candidates with explicit transition diagnostics.

**Tech Stack:** Python, NumPy, pandas, scikit-learn neighbours/clustering/graph tools, LightGBM, XGBoost, existing `src.spatial`, `src.transductive_spatial`, `src.validation`, and `src.validate`.

---

## Current State To Resume

- Public incumbent: `submissions/19_loo_spatial_final.csv` and `submissions/23_loo_spatial_star_tilt.csv`, both `0.96970`.
- Target: exceed `0.97`; remaining public lift is `+0.00030`.
- Best honest OOF remains `submissions/16_spatial_blend.csv` at `0.9690706512708674`; final-only LOO variants are public-risk and have no honest OOF.
- Do not submit:
  - `21_loo_spatial_galaxy_lean.csv`: public evidence does not want more GALAXY.
  - `24_loo_spatial_stronger_nongal.csv`: `22` disproved the lower-GALAXY direction.
  - `25_loo_spatial_xgb_final.csv`: too GALAXY-heavy.
  - `26_loo_spatial_xgb_calibrated.csv`: public `0.96956`, below incumbent.

## Learnings

- Spatial neighbourhood features are the main breakthrough. Raw `alpha`/`delta` underuse a synthetic spatial label structure; kNN class-fraction features lifted the project from the `0.966` band to the `0.969` band.
- The final train/test feature-density mismatch matters. Training LightGBM on leave-one-out spatial train features and predicting test with full-train spatial features improved public score from `0.96927` to `0.96970`.
- Honest OOF and public-risk final candidates must be tracked separately. The best public candidate has no honest OOF because it trains on final-only LOO features.
- Residual classifiers did not work. Graph/cluster residuals scored `0.968377` standalone and got blend weight `0.0`; GALAXY-only residuals selected no flips.
- Class-count steering is bracketed. More GALAXY hurt or was not worth probing; too little GALAXY also hurt. The public-good band is near `GALAXY ~= 156.5k`, `QSO ~= 51.3k-51.4k`, `STAR ~= 39.5k`.
- The current cached probability files are saturated. `19`, `20`, `22`, `23`, and `26` show that simple multiplier and same-cache blend movement is not enough for the remaining `+0.00030`.

## Task 1: Build A Real Graph Label-Propagation Signal

**Files:**
- Create: `scripts/27_graph_label_propagation.py`
- Create: `tests/test_graph_label_propagation.py`
- Output: `experiments/27_graph_label_propagation.json`

- [ ] Write tests for graph propagation on a tiny fixture.

```python
def test_validation_labels_are_hidden_during_oof_propagation():
    # Build a graph where one validation row has a unique label.
    # Flip that row's own label and verify its propagated OOF probabilities do not change.
    # Verify another row can change when the flipped row is in that other row's training graph.
```

- [ ] Implement a kNN graph over train plus test coordinates on unit-sphere `(alpha, delta)`.
- [ ] For OOF evaluation, hide validation labels but keep validation nodes as unlabeled graph nodes.
- [ ] Propagate train labels to unlabeled nodes with a fixed number of random-walk smoothing iterations.
- [ ] Save OOF/test propagation probabilities as `experiments/27_graph_label_propagation_{oof,test}_probabilities.npy`.
- [ ] Acceptance gate: propagated probabilities must add at least `+0.00015` OOF when blended with `16_spatial_blend`, or produce a final-only test candidate with a transition profile not matching failed `22`/`26`.

## Task 2: Add Local Photometric-Neighbour Features

**Files:**
- Create: `scripts/28_local_photometric_neighbours.py`
- Create: `tests/test_local_photometric_neighbours.py`
- Output: `experiments/28_local_photometric_neighbours.json`

- [ ] Build neighbour rates in feature spaces that combine spatial coordinates with photometry, not redshift-only distance.
- [ ] Try at least three spaces:
  - unit-sphere only plus colour vector `[u_g, g_r, r_i, i_z]`;
  - unit-sphere plus magnitudes `[u, g, r, i, z]`;
  - unit-sphere plus redshift and colour, with redshift scale limited so it cannot dominate.
- [ ] Use KFold-OOF train features and full-train test features.
- [ ] Acceptance gate: keep only feature spaces that improve honest OOF or materially decorrelate predictions while preserving the `19` class-count band.

## Task 3: Refit Final LOO LightGBM As A Small Ensemble Family

**Files:**
- Create: `scripts/29_loo_lgbm_family.py`
- Output: `experiments/29_loo_lgbm_family.json`

- [ ] Train a small family of final-only LOO LightGBM models:
  - reference params from `19`;
  - shallower regularized params;
  - deeper regularized params.
- [ ] Average within each family over at least seeds `[42, 43, 44, 45, 46]`.
- [ ] Create calibrated final candidates whose class counts stay within the public-good band:
  - GALAXY between `156450` and `156650`;
  - QSO between `51250` and `51450`;
  - STAR between `39400` and `39600`.
- [ ] Submit at most one candidate from this task unless transition diagnostics are clearly different from `19`/`23`.

## Task 4: Build A Submission-Selection Report Before Any Upload

**Files:**
- Create: `experiments/next_submission_report.md`

- [ ] Compare every new candidate against `19`, `23`, and `16`.
- [ ] Include class counts, changed-row count, transition table, source probability file, multiplier vector, and whether an honest OOF proxy exists.
- [ ] Choose one primary upload and one backup upload.
- [ ] Stop if the best candidate is only a multiplier-only variant of existing `19` or `25` cached probabilities.

## Verification Gates For Tomorrow

- `uv run pytest -q`
- `uv run ruff check .`
- `uv run python -m src.validate <candidate.csv>`
- Update `experiments/leaderboard.md`, `experiments/candidate_recommendations.md`, `PROGRESS.md`, `DECISIONS.md`, and `tasks/lessons.md` immediately after any public submission.

## Stop Conditions

- Stop public submissions for the day if two new candidates score below `0.96970`.
- Stop multiplier-only probing from a cache after one regression in each direction.
- Do not declare a new local ceiling while the public leaderboard still shows a higher cluster; instead identify what signal the current representation is missing.
