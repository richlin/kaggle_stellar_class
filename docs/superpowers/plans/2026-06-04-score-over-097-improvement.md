# Score Over 0.97 Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the Kaggle public balanced-accuracy score above `0.97` from the current `03_final.csv` public score of `0.96691`.

**Architecture:** Keep the current LightGBM CV/threshold pipeline as the reference model, then add score-improving diversity through probability blending, targeted error analysis, and constrained threshold search. Every candidate must produce OOF probabilities, test probabilities, a tracked experiment record, and a valid submission before it is considered for public submission.

**Tech Stack:** Python, pandas, numpy, scikit-learn, LightGBM, **XGBoost and CatBoost as cross-library base learners** (installed via `uv pip install xgboost catboost scipy`; fall back to LightGBM-only blend if either fails), `scipy.optimize` for continuous threshold search, pytest, ruff.

---

## Plan Revisions — 2026-06-04 (post-confusion-analysis)

After running the OOF confusion analysis (see "Current Baseline → Error Analysis" below), four modifications were agreed before implementation:

1. **Lead with cross-library diversity (XGBoost + CatBoost), not LightGBM variants.** LightGBM-variant blends (dart/extra_trees/deep) are weakly decorrelated — same library, same histogram split algorithm — so their blend gain is small. Real decorrelation at the GALAXY↔STAR boundary comes from different families. XGBoost (different split/regularization) and CatBoost (ordered target encoding on the soft categoricals) become first-class base learners; the LightGBM variants collapse to **one optional candidate (`lgbm_dart`)**.
2. **Add a continuous threshold optimizer.** Replace reliance on the coarse 6-value multiplier grid with a `scipy.optimize` (Nelder-Mead) search over the 3 class multipliers on blended OOF, **keeping the existing fold- and class-recall stability guards** so we do not overfit OOF. Near-free expected gain of +0.001–0.002.
3. **Reuse the completed error analysis.** Task 5's confusion/recall step is already done (results below); do not re-derive it. Any boundary features (Task 6) must target **low-redshift GALAXY/STAR separation specifically**, not generic expansion.
4. **Extend graceful dependency fallback to XGBoost** (same posture the plan already had for CatBoost — attempt install, do not fight dependency issues).

---

## Current Baseline

- Official public score: `0.96691` from `submissions/03_final.csv`.
- Local repeated-seed tuned OOF: `0.9659249816190973`.
- Current final model: 3-seed LightGBM probability average with stable class multipliers `[0.9, 0.8, 1.15]`.
- Target gap: approximately `+0.0031` public score.

### Error Analysis (completed 2026-06-04 — Task 5 Step 1)

OOF confusion on the 3-seed final model (`experiments/03_final_oof_probabilities.npy`, multipliers `[0.9, 0.8, 1.15]`), rows = true, cols = pred:

|        | GALAXY | QSO    | STAR  |
|--------|--------|--------|-------|
| GALAXY | 361236 | 4747   | 11497 |
| QSO    | 2029   | 113854 | 1260  |
| STAR   | 2251   | 323    | 80150 |

Per-class recall: GALAXY `0.957`, QSO `0.972`, STAR `0.969`. Balanced accuracy `0.96592`.

Key findings driving the strategy:

- **GALAXY is the bottleneck class** (lowest recall) *and* the majority class. Because balanced accuracy is macro-recall, lifting GALAXY recall has outsized leverage.
- **~84% of the error budget is in the GALAXY row:** GALAXY→STAR (`11,497`) and GALAXY→QSO (`4,747`) dominate.
- The leak is a **low-redshift boundary problem**: STAR redshift is tightly ≈0 (75th pct `0.10`), and low-redshift GALAXYs overlap it; the categorical columns (`spectral_type`, `galaxy_population`) are *soft* signals, not deterministic separators.

## Strategy

The highest-probability route is **cross-library model diversity** plus a **continuous balanced-accuracy threshold optimizer**, not another small single-LightGBM parameter tweak. The current model is already close to the ceiling for one LightGBM family, and LightGBM-variant blends are weakly decorrelated. Train XGBoost and CatBoost on the identical fold splits, blend OOF probabilities with grid-searched weights, then optimize the class multipliers continuously (with stability guards) on the blended OOF. Boundary feature work is reserved for *after* the ensemble and must target the low-redshift GALAXY/STAR boundary identified above.

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

```python
def test_continuous_threshold_search_beats_or_matches_coarse_grid():
    # On toy OOF probabilities the continuous (Nelder-Mead) multiplier search must return
    # a balanced accuracy >= the coarse-grid search, and must respect the same
    # fold/class-recall stability guards (no guarded regression beyond the limits).
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
    # For every blend, run the threshold multiplier search.
    # Return best weights, multipliers, OOF score, and per-class recall.
```

Also implement the continuous threshold optimizer (Modification 2). It wraps the existing
stability logic so OOF is not overfit:

```python
def search_continuous_multipliers(y_true, probabilities, fold_ids, class_labels):
    # Use scipy.optimize.minimize (Nelder-Mead) over the 3 class multipliers to MAXIMIZE
    # balanced accuracy on OOF (minimize the negative). Seed from the coarse-grid optimum
    # so we never do worse than the grid. Reject any optimum that violates the existing
    # MATERIAL_FOLD_REGRESSION / MATERIAL_CLASS_RECALL_REGRESSION stability guards,
    # falling back to the coarse-grid result. Return multipliers, score, per-class recall.
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_ensemble.py -q
```

Expected: pass.

## Task 3: Add Cross-Library Base Learners (XGBoost + one LightGBM variant)

> **Modification 1:** Lead with cross-library diversity. XGBoost is the primary new learner; `lgbm_dart` is the only retained LightGBM variant (extra_trees / deep_regularized are dropped — too correlated with the existing LightGBM to earn their compute). CatBoost follows in Task 4.

**Files:**
- Modify: `scripts/04_ensemble.py`
- Output: `experiments/04_*_oof_probabilities.npy`, `experiments/04_*_test_probabilities.npy`
- Output: appended records in `experiments/runs.jsonl`

- [ ] **Step 1: Reuse existing final probabilities**

Load existing arrays:

```python
experiments/03_final_oof_probabilities.npy
experiments/03_final_test_probabilities.npy
```

Treat them as model `lgbm_seed_average_final` (the reference base learner — do not retrain).

- [ ] **Step 2: Train XGBoost on the identical fold splits**

XGBoost is the primary diversity source (different split-finding and regularization than LightGBM).
Note: XGBoost has no native pandas-category handling here, so one-hot or integer-encode the
categorical columns inside the harness (keep the encoding deterministic and shared train/test).

```python
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 1200,
    "learning_rate": 0.04,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "tree_method": "hist",
    "n_jobs": -1,
}
# Apply per-sample class weights (balanced) since XGBoost has no class_weight arg.
```

- [ ] **Step 3: Train one diverse LightGBM variant (`lgbm_dart`)**

Retain a single LightGBM variant for cheap intra-library diversity:

```python
LGBM_DART = {
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
}
```

- [ ] **Step 4: Train with the same fold protocol**

For each candidate (same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` as the
reference model so OOF rows align for blending):

```bash
uv run python scripts/04_ensemble.py --train-candidate <candidate-name>
```

Each run must save:

```text
experiments/04_<candidate>_oof_probabilities.npy
experiments/04_<candidate>_test_probabilities.npy
```

And append candidate score details to `experiments/runs.jsonl`.

- [ ] **Step 5: Accept or reject each candidate**

Keep a candidate for blending if either:

- Its OOF score is competitive with the current final, or
- Its predictions are diverse enough to improve blend OOF even if standalone score is lower
  (check pairwise prediction-disagreement against the reference model).

Reject candidates that are both lower-scoring and do not improve any tested blend.

## Task 4: Add CatBoost (Modification 4: graceful fallback, same as XGBoost)

**Files:**
- Modify: `requirements.txt` to pin `xgboost`, `catboost`, `scipy` once installs succeed
- Modify: `scripts/04_ensemble.py`
- Output: `experiments/04_catboost_*.npy`

- [ ] **Step 1: Confirm install feasibility (both libraries)**

Installed via:

```bash
VIRTUAL_ENV=.venv uv pip install xgboost catboost scipy
```

If either install fails or disrupts the environment, drop that learner and continue with whatever
base learners are available (down to LightGBM-only). Do not fight dependency issues.

- [ ] **Step 2: Add CatBoost CV candidate**

CatBoost uses the soft categoricals natively (ordered target encoding — its main source of
diversity here):

```python
CatBoostClassifier(
    loss_function="MultiClass",
    iterations=1200,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=5,
    random_seed=seed,
    auto_class_weights="Balanced",
    verbose=False,
    allow_writing_files=False,
)
```

Train with the same 5-fold split discipline and save OOF/test probabilities.

- [ ] **Step 3: Blend across families**

Search weights for:

- reference LightGBM only
- + XGBoost
- + CatBoost
- + `lgbm_dart`
- best 3- and 4-model combination

Keep each learner only if it improves the OOF blend over the best blend without it.

## Task 5: Run Targeted Error Analysis

**Files:**
- Create: `scripts/05_error_analysis.py`
- Output: `experiments/05_error_analysis.json`

- [x] **Step 1: Analyze OOF confusion and confidence** — *partially complete (Modification 3)*

The confusion matrix, per-class recall, and dominant error directions are already recorded under
"Current Baseline → Error Analysis": the leak is **GALAXY→STAR (`11,497`) and GALAXY→QSO (`4,747`)**,
a low-redshift boundary. `scripts/05_error_analysis.py` only needs to add, on top of that:

- Confidence margin for correct vs incorrect predictions.
- Feature summaries for the lowest-margin GALAXY errors (focus on `redshift`, colors, and the
  `spectral_type` / `galaxy_population` categoricals at low redshift).

Skip re-deriving the confusion matrix — reuse the table above.

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

- [ ] **Step 2: Re-tune class multipliers (continuous — Modification 2)**

For the best blend:

- run `search_continuous_multipliers` (Nelder-Mead, seeded from the coarse-grid optimum) on the
  blended OOF probabilities
- enforce the existing fold and class-recall stability guards; fall back to the coarse-grid result
  if the continuous optimum violates them
- record argmax score, coarse-grid score, continuous-tuned score, per-class recall, and chosen
  multipliers (so the continuous gain over the grid is auditable)

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
2. Build and test ensemble helpers (blend + **continuous threshold optimizer**).
3. Train cross-library learners: **XGBoost** (primary) and **`lgbm_dart`** on identical fold splits.
4. Add **CatBoost** (graceful fallback applies to both XGBoost and CatBoost).
5. Blend probabilities, run the continuous threshold optimizer, and submit the best evidence-backed ensemble.
6. If the ensemble does not clear `0.97`, extend error analysis (margins on the already-known GALAXY→STAR/QSO leak).
7. Add focused **low-redshift GALAXY/STAR boundary** features only when that analysis justifies them.
8. Use pseudo-labeling only as a final, carefully constrained experiment.
