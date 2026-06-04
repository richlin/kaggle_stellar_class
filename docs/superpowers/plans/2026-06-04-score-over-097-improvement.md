# Score Over 0.97 Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the Kaggle public balanced-accuracy score above `0.97` from the current `03_final.csv` public score of `0.96691`.

**Architecture:** Keep the current LightGBM CV/threshold pipeline as the reference model, then add score-improving diversity through probability blending, targeted error analysis, and constrained threshold search. Every candidate must produce OOF probabilities, test probabilities, a tracked experiment record, and a valid submission before it is considered for public submission.

**Tech Stack:** Python, pandas, numpy, scikit-learn, LightGBM, optional CatBoost only if it installs cleanly through the project environment, pytest, ruff.

---

## Current Baseline

- Official public score: `0.96691` from `submissions/03_final.csv`.
- Local repeated-seed tuned OOF: `0.9659249816190973`.
- Current final model: 3-seed LightGBM probability average with stable class multipliers `[0.9, 0.8, 1.15]`.
- Target gap: approximately `+0.0031` public score.

## Strategy

The highest-probability route is model diversity, not another small single-LightGBM parameter tweak. The current model is already close to the local/public ceiling for one LightGBM family. The next work should test whether different learners or sufficiently different LightGBM configurations make complementary errors, then blend OOF probabilities and re-tune class multipliers.

## Task 1: Record Official Score And Reproduce Current Best

**Files:**
- Modify: `experiments/leaderboard.md`
- Modify: `PROGRESS.md`
- Read: `experiments/03_tune.json`
- Verify: `submissions/03_final.csv`

- [ ] **Step 1: Validate the currently submitted file**

Run:

```bash
uv run python -m src.validate submissions/03_final.csv
```

Expected:

```text
OK: submissions/03_final.csv is a valid submission
```

- [ ] **Step 2: Record the official public score**

Update `experiments/leaderboard.md` so the `submissions/03_final.csv` row has:

```markdown
| 2026-06-04 | `submissions/03_final.csv` | Repeated-seed tuned OOF balanced accuracy | 0.965925 | 0.96691 | 0.000985 | Phase 4 final candidate; official public score from Kaggle. |
```

- [ ] **Step 3: Add the new improvement target to progress**

Add to `PROGRESS.md`:

```markdown
- 2026-06-04: `submissions/03_final.csv` official public score: `0.96691`.
  - New target: exceed `0.97` public balanced accuracy.
  - Required public lift: approximately `+0.0031`.
```

- [ ] **Step 4: Run verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
```

Expected: all tests pass and Ruff reports no errors.

## Task 2: Build Ensemble Harness

**Files:**
- Create: `scripts/04_ensemble.py`
- Create: `tests/test_ensemble.py`
- Modify: `src/validation.py` only if a reusable blending helper is needed
- Output: `experiments/04_ensemble.json`
- Output: `submissions/04_ensemble.csv`

- [ ] **Step 1: Write failing tests for probability blending**

Create `tests/test_ensemble.py` with tests for:

```python
def test_weighted_blend_preserves_shape_and_normalizes_weights():
    # Two model probability arrays of shape (2, 3).
    # Weights [2, 1] should behave like normalized weights [2/3, 1/3].
    # Result shape must stay (2, 3).
```

```python
def test_search_blend_weights_prefers_better_oof_blend():
    # Toy y_true plus two probability arrays.
    # The helper should choose the blend with highest balanced accuracy after threshold tuning.
```

```python
def test_make_ensemble_submission_preserves_id_order():
    # Encoded predictions from blended test probabilities must decode to class labels
    # and preserve sample_submission id order.
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_ensemble.py -q
```

Expected: fails because `scripts/04_ensemble.py` does not exist or helper functions are missing.

- [ ] **Step 3: Implement minimal blending helpers**

In `scripts/04_ensemble.py`, implement:

```python
def weighted_probability_blend(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    blended = np.zeros_like(probabilities[0], dtype=float)
    for probability, weight in zip(probabilities, weights, strict=True):
        blended += probability * weight
    return blended
```

Also implement:

```python
def search_blend_weights(y_true, probability_sets, fold_ids, class_labels):
    # Search small grids first:
    # - two-model grid: 0.0, 0.1, ..., 1.0
    # - three-model grid: all weights in 0.1 increments summing to 1.0
    # For every blend, run stable multiplier search.
    # Return best weights, multipliers, OOF score, and per-class recall.
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_ensemble.py -q
```

Expected: pass.

## Task 3: Add Diverse LightGBM Candidates

**Files:**
- Modify: `scripts/04_ensemble.py`
- Output: `experiments/04_lgbm_*.npy` probability arrays
- Output: appended records in `experiments/runs.jsonl`

- [ ] **Step 1: Reuse existing final probabilities**

Load existing arrays:

```python
experiments/03_final_oof_probabilities.npy
experiments/03_final_test_probabilities.npy
```

Treat them as model `lgbm_seed_average_final`.

- [ ] **Step 2: Train diverse LightGBM configs**

Add at least three genuinely different LightGBM probability producers:

```python
LGBM_DIVERSE_CANDIDATES = [
    {
        "name": "lgbm_dart",
        "params": {
            "boosting_type": "dart",
            "objective": "multiclass",
            "class_weight": "balanced",
            "n_estimators": 900,
            "learning_rate": 0.035,
            "num_leaves": 47,
            "min_child_samples": 40,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.9,
            "lambda_l1": 0.05,
            "lambda_l2": 0.2,
        },
    },
    {
        "name": "lgbm_extra_trees",
        "params": {
            "objective": "multiclass",
            "class_weight": "balanced",
            "extra_trees": True,
            "n_estimators": 900,
            "learning_rate": 0.035,
            "num_leaves": 63,
            "min_child_samples": 30,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.8,
            "lambda_l2": 0.5,
        },
    },
    {
        "name": "lgbm_deep_regularized",
        "params": {
            "objective": "multiclass",
            "class_weight": "balanced",
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 95,
            "min_child_samples": 80,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
        },
    },
]
```

- [ ] **Step 3: Train with the same fold protocol**

For each candidate:

```bash
uv run python scripts/04_ensemble.py --train-candidate <candidate-name>
```

Each run must save:

```text
experiments/04_<candidate>_oof_probabilities.npy
experiments/04_<candidate>_test_probabilities.npy
```

And append candidate score details to `experiments/runs.jsonl`.

- [ ] **Step 4: Accept or reject each LightGBM candidate**

Keep a candidate for blending if either:

- Its OOF score is competitive with current final, or
- Its predictions are diverse enough to improve blend OOF even if standalone score is lower.

Reject candidates that are both lower-scoring and do not improve any tested blend.

## Task 4: Try CatBoost If Dependency Install Is Clean

**Files:**
- Modify: `requirements.txt` only if CatBoost installation is approved and succeeds
- Modify: `scripts/04_ensemble.py`
- Output: `experiments/04_catboost_*.npy`

- [ ] **Step 1: Check install feasibility**

Run:

```bash
uv add catboost
```

If installation fails or significantly disrupts the environment, stop CatBoost work and continue with LightGBM-only ensemble. Do not spend time fighting dependency issues.

- [ ] **Step 2: Add CatBoost CV candidate if installed**

Use native categorical columns:

```python
CatBoostClassifier(
    loss_function="MultiClass",
    iterations=1200,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=5,
    random_seed=seed,
    verbose=False,
    allow_writing_files=False,
)
```

Train with the same 5-fold split discipline and save OOF/test probabilities.

- [ ] **Step 3: Blend CatBoost with LightGBM**

Search weights for:

- current final LightGBM only
- LightGBM diverse candidates
- CatBoost
- CatBoost + best LightGBM blend

Keep CatBoost only if OOF blend improves over the LightGBM-only blend.

## Task 5: Run Targeted Error Analysis

**Files:**
- Create: `scripts/05_error_analysis.py`
- Output: `experiments/05_error_analysis.json`

- [ ] **Step 1: Analyze OOF confusion and confidence**

Load:

```text
experiments/03_final_oof_probabilities.npy
experiments/03_tune.json
```

Compute:

- Confusion matrix.
- Per-class recall.
- Error counts for `GALAXY -> STAR`, `GALAXY -> QSO`, `QSO -> GALAXY`, `STAR -> GALAXY`.
- Confidence margin for correct vs incorrect predictions.
- Feature summaries for the lowest-margin errors.

- [ ] **Step 2: Identify boundary-specific feature candidates**

Only test features tied to actual OOF mistakes. Candidate families:

- `delta_sin`, `delta_cos`
- 3D sky unit vector from `alpha` and `delta`
- `redshift_squared`, `log1p_redshift`
- redshift quantile bins
- more color-redshift interactions
- color ratio or slope features
- class-centroid distance features for photometric colors
- `spectral_population` x redshift-bin interaction

- [ ] **Step 3: Prioritize by expected gain**

Only promote feature families to modeling if the error analysis shows a plausible affected boundary. Do not add broad feature expansion just because it is easy.

## Task 6: Test Focused Feature Additions

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`
- Modify: model scripts if feature set names are added

- [ ] **Step 1: Add feature-set support**

Extend:

```python
build_feature_frame(df, feature_set="baseline")
```

to support:

```python
feature_set="boundary_v1"
```

The default remains `"baseline"`.

- [ ] **Step 2: Add tests first**

Add tests proving:

- baseline feature columns remain unchanged
- boundary feature columns exist for `feature_set="boundary_v1"`
- train/test columns match
- no NaNs or infinities are introduced
- categorical columns remain categorical

- [ ] **Step 3: Run OOF check**

Run a single-seed 5-fold LightGBM with `boundary_v1`.

Accept `boundary_v1` only if:

- OOF improves over baseline feature set, or
- It improves a weak class recall without materially lowering balanced accuracy, or
- It improves ensemble blend OOF.

## Task 7: Blend And Submit Candidate

**Files:**
- Modify: `scripts/04_ensemble.py`
- Output: `experiments/04_ensemble.json`
- Output: `submissions/04_ensemble.csv`
- Modify: `experiments/leaderboard.md`

- [ ] **Step 1: Search blend weights on OOF probabilities**

Use only saved OOF probability arrays. Never tune weights on public score.

Search:

- all 2-model blends
- all 3-model blends
- best 4-model blend if three-model blend improves

- [ ] **Step 2: Re-tune stable class multipliers**

For the best blend:

- tune multipliers on blended OOF probabilities
- enforce fold and class recall stability
- record argmax score, tuned score, per-class recall, and chosen multipliers

- [ ] **Step 3: Write `submissions/04_ensemble.csv`**

Validate:

```bash
uv run python -m src.validate submissions/04_ensemble.csv
wc -l submissions/04_ensemble.csv
```

Expected row count:

```text
247436 submissions/04_ensemble.csv
```

- [ ] **Step 4: Submit only if evidence justifies it**

Submit if:

- OOF score beats `0.9659249816190973`, and
- the blend has credible model diversity, and
- no class recall regresses enough to erase the balanced-accuracy gain.

If the OOF gain is small but model diversity is high, submit one carefully chosen candidate. Avoid repeated public leaderboard probing.

## Task 8: Pseudo-Labeling Only After Ensemble

**Files:**
- Create: `scripts/06_pseudolabel.py` only if ensemble still falls short

- [ ] **Step 1: Use only high-confidence ensemble predictions**

Candidate thresholds:

- max probability >= `0.995`
- prediction margin >= `0.75`

- [ ] **Step 2: Train with pseudo-labels as low-weight rows**

Do not fully trust pseudo-labels. If supported by the model API, down-weight pseudo-labeled test rows relative to true training rows.

- [ ] **Step 3: Accept only if OOF proxy and public result agree**

Pseudo-labeling cannot be honestly validated on the test set, so use it sparingly and only after stronger ensemble work.

## Verification Gates

Before any new public submission:

```bash
uv run pytest -q
uv run ruff check .
uv run python -m src.validate submissions/<candidate>.csv
wc -l submissions/<candidate>.csv
```

Every score-producing run must have:

- a JSON or JSONL experiment record
- OOF score
- per-class recall
- chosen multipliers
- submission path
- public score recorded after submission

## Recommended Execution Order

1. Record current official score.
2. Build and test ensemble helpers.
3. Add diverse LightGBM candidates.
4. Add CatBoost only if installation succeeds cleanly.
5. Blend probabilities and submit the best evidence-backed ensemble.
6. Run targeted error analysis if ensemble does not clear `0.97`.
7. Add focused boundary features only when error analysis justifies them.
8. Use pseudo-labeling only as a final, carefully constrained experiment.
