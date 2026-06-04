# Implementation Plan: Predicting Stellar Class (Kaggle)

> **For agentic workers:** use `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` to execute this plan task-by-task.
> Keep `tasks/todo.md`, `PROGRESS.md`, and `DECISIONS.md` in sync with completed work.

## Context

Kaggle competition: classify each astronomical object as `GALAXY`, `QSO`, or `STAR`.
**Metric: balanced accuracy** (= mean per-class recall), which makes the 65/20/14 class
imbalance the central challenge — a model that favors the majority GALAXY class scores poorly.

Current state: only raw data exists (`raw_data/{train,test,sample_submission}.csv`). No code
or project environment is configured. Local probe found `python3`, `pandas 2.3.3`, and
`numpy 2.0.2`; `scikit-learn` and `lightgbm` are not installed yet, and `python` is not
currently on PATH.

**Data facts** (from exploration of `train.csv`, 577,347 rows):
- Numeric features: `alpha, delta` (sky coords), `u, g, r, i, z` (photometric magnitudes), `redshift`.
- Categorical features: `spectral_type` (M, G/K, A/F, O/B), `galaxy_population` (Red_Sequence, Blue_Cloud).
- Target: `class`. Distribution: GALAXY 65.4%, QSO 20.3%, STAR 14.3%.
- No missing values, no duplicate ids, no unseen categorical levels in test, and
  `sample_submission.csv` ids match `test.csv` exactly.
- Categoricals are **informative but not perfectly leaky** (each value spans all 3 classes),
  but the class balance inside each category is highly uneven and useful.
- `redshift` is the dominant signal (STAR≈0.07, GALAXY≈0.51, QSO≈1.88 mean) but ranges overlap.
- A rough redshift-only threshold rule scored about 0.839 balanced accuracy on train, so
  high-score work should focus on ambiguous GALAXY/QSO/STAR overlap zones rather than basic
  separability.
- `test.csv` (247,435 rows) has the same features minus `class`. Submission format: `id,class`.

**Chosen strategy** (confirmed with user): ship a fast LightGBM baseline first, then iterate
with feature engineering, cross-validation, and per-class threshold tuning to maximize balanced accuracy.

## Architecture Decisions

- **Model path: LightGBM first, model diversity later** — LightGBM is fast on 577k rows,
  handles categoricals, and gives a strong baseline. Add CatBoost or a second gradient
  boosting model only after the CV/threshold harness is reproducible.
- **Imbalance handling: balanced class weights during training + per-class threshold/prior
  adjustment on out-of-fold predictions** to directly optimize balanced accuracy. This is the
  highest-leverage lever for this metric.
- **Validation: stratified K-fold (5) with OOF probabilities, then repeated-seed confirmation
  for final candidates.** Balanced accuracy must be estimated on OOF predictions, never on
  training data. Threshold tuning on the training fit would overfit.
- **Experiment ledger is required.** Every run must record feature set, params, seed, fold
  scores, OOF balanced accuracy, per-class recall, threshold vector, submission path, and
  Kaggle public score if submitted.
- **Code layout: a thin shared module + one script per phase**, so every phase produces a
  scoreable submission and stays debuggable (per user's "simplicity above all").
  - `src/data.py` — load CSVs, build features, encode categoricals (shared by all phases).
  - `src/features.py` — deterministic feature builders grouped by feature family.
  - `src/validation.py` — metrics, submission validation, OOF threshold search, experiment logging.
  - `scripts/01_baseline.py` — Phase 1 holdout baseline → submission.
  - `scripts/02_cv_threshold.py` — Phases 2–3 K-fold + threshold tuning → submission.
  - `scripts/03_tune.py` — Phase 4 hyperparameter tuning → final submission.
  - `scripts/04_ensemble.py` — Phase 5 optional model diversity / probability averaging.
  - `tests/` — focused pytest coverage for feature generation, validation, and submission format.
  - `submissions/` — output CSVs, named per phase.
  - `experiments/` — run ledger files and OOF/test probability artifacts.

## Task List

### Phase 0: Setup
- **Task 0 (S):** Create env + scaffolding.
  - Create conda env (or venv) and install `pandas numpy scikit-learn lightgbm pytest`.
  - Pin versions in `requirements.txt`.
  - Create dirs `src/`, `scripts/`, `tests/`, `submissions/`, `experiments/`, `tasks/`.
  - Write `tasks/todo.md` from this plan, plus root `PROGRESS.md` and `DECISIONS.md`.
  - **Acceptance:** `.venv/bin/python -c "import pandas, sklearn, lightgbm, pytest"` succeeds.
  - **Verification:** prints library versions without error.

- **Task 0.5 (S):** Add data-contract checks before modeling.
  - Create `tests/test_data_contract.py`.
  - Verify train/test/sample row counts, unique ids, sample ids equal test ids, required
    columns exist, target labels are exactly `{GALAXY,QSO,STAR}`, no missing values, and test
    categorical levels are a subset of train levels.
  - **Acceptance:** data-contract tests pass before any model code is written.
  - **Verification:** `.venv/bin/pytest tests/test_data_contract.py -q`.

### Phase 1: End-to-end baseline (first scoreable submission)
- **Task 1 (M):** `src/data.py` + `src/features.py` — shared data and feature layer.
  - `load_raw()` reads train/test from `raw_data/`.
  - `build_features(df, feature_set="baseline")` keeps raw numerics and casts
    `spectral_type` + `galaxy_population` to pandas `category` dtype.
  - Baseline feature set includes:
    - Adjacent colors: `u_g, g_r, r_i, i_z`.
    - Wider colors: `u_r, u_i, u_z, g_i, g_z, r_z`.
    - Magnitude summary features across `u,g,r,i,z`: mean, std, min, max, range.
    - Coordinate encoding: `alpha_sin, alpha_cos` plus raw `alpha, delta`.
    - Redshift interactions: `redshift_x_u_g, redshift_x_g_r, redshift_x_r_i, redshift_x_i_z`.
    - Simple categorical interaction: `spectral_population = spectral_type + "__" + galaxy_population`.
  - Returns feature matrix `X`, target `y` (label-encoded GALAXY/QSO/STAR), and the categorical
    column names.
  - **Acceptance:** returns `X` with expected columns, no NaNs introduced by feature math,
    train/test columns match exactly, and label encoder round-trips (`inverse_transform`
    recovers original strings).
  - **Verification:** `.venv/bin/pytest tests/test_features.py -q`; a `__main__` block prints
    `X.shape`, dtypes, and `y` value counts for manual inspection.
- **Task 2 (M):** `scripts/01_baseline.py` — train + submit.
  - Stratified 80/20 holdout. Train `LGBMClassifier(class_weight="balanced", objective="multiclass")`
    with sensible defaults (e.g. `n_estimators=300, learning_rate=0.05`).
  - Report `balanced_accuracy_score` + per-class recall + confusion matrix on the holdout.
  - Refit on full train, predict test, write `submissions/01_baseline.csv` with columns `id,class`.
  - Save `experiments/01_baseline.json` with params, feature list, seed, holdout score, per-class
    recall, submission path, and timestamp.
  - **Acceptance:** holdout balanced accuracy printed (sanity target ≳ 0.90 given strong signal);
    submission has exactly 247,435 rows + header, columns `id,class`, labels ∈ {GALAXY,QSO,STAR}.
  - **Verification:** `.venv/bin/pytest tests/test_submission.py -q`; `wc -l
    submissions/01_baseline.csv` → 247436; `id` order matches `sample_submission.csv` exactly.

### Checkpoint A — after Phase 1
- [ ] `submissions/01_baseline.csv` is valid and matches sample submission ids/format.
- [ ] Holdout balanced accuracy recorded as the baseline number to beat.
- [ ] **Review with user before proceeding** (optionally submit to Kaggle to confirm public LB ≈ local).

### Phase 2: Cross-validated training (robust estimate)
- **Task 3 (M):** `scripts/02_cv_threshold.py` (CV part).
  - 5-fold `StratifiedKFold`; train LightGBM per fold with `class_weight="balanced"` and early
    stopping on the fold's validation set; collect **OOF predicted probabilities** for all train rows.
  - Print OOF balanced accuracy (argmax) — the honest estimate to optimize against.
  - Print per-fold balanced accuracy and per-fold per-class recall; flag any class whose recall
    varies by more than 0.02 across folds.
  - Average the 5 fold models' test-set probabilities (bagged prediction).
  - Save OOF probabilities and averaged test probabilities under `experiments/02_cv_*`.
  - **Acceptance:** OOF balanced accuracy ≥ Phase 1 holdout number; OOF prob matrix shape
    `(577347, 3)`; fold-level recall is stable enough to trust the estimate.
  - **Verification:** printed OOF balanced accuracy + per-class recall + fold table;
    `.venv/bin/pytest tests/test_validation.py -q`.

### Phase 3: Threshold / prior tuning (the metric lever)
- **Task 4 (M):** extend `scripts/02_cv_threshold.py` (tuning part).
  - On OOF probabilities, search a per-class multiplier vector `w` applied before argmax
    (`pred = argmax(proba * w)`) to **maximize OOF balanced accuracy** — coordinate-ascent over
    a small grid is sufficient and cheap. Equivalent intuition: down-weight the GALAXY prior.
  - Validate threshold stability by fitting `w` on 4 folds' OOF predictions and evaluating on
    the held-out fold's OOF predictions. Reject threshold vectors that improve global OOF but
    hurt one class or one fold materially.
  - Apply the learned `w` to the averaged test probabilities → `submissions/02_cv_tuned.csv`.
  - Record the untuned score, tuned score, per-class recall before/after, chosen `w`, and
    threshold-stability results in `experiments/02_cv_threshold.json`.
  - **Acceptance:** tuned OOF balanced accuracy > untuned OOF balanced accuracy and no class recall
    regresses enough to erase the balanced-accuracy gain; submission valid.
  - **Verification:** print before/after OOF balanced accuracy and the chosen `w`; validate
    submission row count/format as in Task 2.

### Checkpoint B — after Phase 3
- [ ] Tuned OOF balanced accuracy beats the Phase 1 baseline.
- [ ] `submissions/02_cv_tuned.csv` valid. **Review with user** (submit and compare to local OOF).
- [ ] `experiments/leaderboard.md` records local OOF score, public LB score, and delta if submitted.

### Phase 4: Hyperparameter tuning + polish
- **Task 5 (L):** `scripts/03_tune.py`.
  - Tune key LightGBM params (`num_leaves`, `learning_rate`, `n_estimators` via early stopping,
    `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`) using
    the Phase 2 CV harness + OOF balanced accuracy as the objective (manual small grid first;
    Optuna only if the user explicitly approves the extra dependency).
  - Re-run threshold tuning (Task 4 logic) on the best model's OOF probs.
  - Run an ablation table for feature families: raw magnitudes, colors, magnitude summaries,
    coordinate encoding, redshift interactions, categorical interaction. Keep only families that
    improve repeated-CV OOF or per-class recall without destabilizing folds.
  - Repeat the best candidate over at least 3 seeds before declaring it better than Phase 3.
  - Produce `submissions/03_final.csv`.
  - **Acceptance:** final repeated-seed OOF balanced accuracy ≥ Phase 3; no overfitting gap blow-up
    (train vs OOF sane); feature changes have ablation evidence.
  - **Verification:** print final OOF balanced accuracy + params + ablation table; submission valid.

### Phase 5: Model diversity / ensemble (high-score push)
- **Task 6 (M/L):** `scripts/04_ensemble.py`.
  - Add CatBoost only if install succeeds cleanly in the project env; otherwise use a second
    LightGBM configuration with different depth/regularization as the diversity model.
  - Train each model through the same 5-fold CV harness and save compatible OOF/test probabilities.
  - Search ensemble weights on OOF probabilities, then re-run per-class threshold tuning on the
    blended OOF probabilities.
  - Write `submissions/04_ensemble.csv`.
  - **Acceptance:** ensemble OOF balanced accuracy beats the best single model on repeated-seed
    validation, or the plan explicitly keeps the simpler single model.
  - **Verification:** print single-model vs ensemble OOF table, ensemble weights, threshold vector,
    and valid submission checks.

### Checkpoint C — Complete
- [ ] All submissions valid and reproducible from scripts.
- [ ] `experiments/runs.csv` or JSONL records every score-producing run.
- [ ] `experiments/leaderboard.md` records submitted files and public scores.
- [ ] `PROGRESS.md` + `DECISIONS.md` updated with scores per phase and rationale.
- [ ] Final local OOF, repeated-seed OOF, public LB score, and chosen submission are documented.
- [ ] Ready for independent review before final Kaggle submission.

## Required Test Coverage

- `tests/test_data_contract.py`
  - Row counts: train 577,347, test 247,435, sample submission 247,435.
  - Unique ids in train and test.
  - Test ids equal sample submission ids in the same order.
  - Required columns exist and target labels are exactly `{GALAXY,QSO,STAR}`.
  - No missing values in train/test.
  - Test categorical levels are covered by train categorical levels.
- `tests/test_features.py`
  - `build_features()` returns identical train/test feature columns.
  - All engineered feature names are present.
  - No NaNs or infinities are introduced by feature math.
  - Categorical columns have pandas `category` dtype.
  - Label encoding round-trips all class names.
- `tests/test_validation.py`
  - Balanced accuracy helper matches `sklearn.metrics.balanced_accuracy_score`.
  - Threshold search preserves output shape and returns one multiplier per class.
  - Threshold tuning on a toy fixture improves or preserves balanced accuracy.
  - Experiment logging writes required keys.
- `tests/test_submission.py`
  - Submission has columns `id,class`.
  - Submission has exactly 247,435 rows.
  - Submission ids match `sample_submission.csv` exactly.
  - Submission labels are limited to `{GALAXY,QSO,STAR}`.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Threshold tuning overfits | Med | Tune only on OOF probabilities; add fold-held-out threshold stability checks. |
| Local OOF ≠ Kaggle LB | Med | Track every submission in `experiments/leaderboard.md`; trust repeated CV over single LB readings. |
| 577k rows × 5 folds slow on laptop | Low | LightGBM is fast; use `n_jobs=-1`, early stopping, moderate `n_estimators`. |
| Python 3.13 wheel availability for lightgbm | Low | Verify install in Task 0; fall back to conda-forge build if pip wheel missing. |
| CatBoost install/runtime cost slows iteration | Low/Med | Keep CatBoost in Phase 5 only; do not block LightGBM baseline or CV harness. |
| Public leaderboard probing causes overfit | Med | Submit only checkpoint candidates; require local OOF evidence before trusting a public LB gain. |
| Feature explosion adds noise | Med | Use feature-family ablations and repeated-seed CV before keeping new feature groups. |

## Open Questions
- Submit to Kaggle at each checkpoint, or only the final? (Plan assumes optional submit at A/B, mandatory awareness at C.)
- Optuna acceptable as a Phase-4 dependency, or keep to a manual grid? (Plan defaults to manual grid unless told otherwise.)
- CatBoost acceptable as a Phase-5 dependency if LightGBM has plateaued? (Plan defaults to optional, not baseline-blocking.)
- Should the final model optimize for public leaderboard rank or most trustworthy local OOF? (Plan defaults to trustworthy local OOF.)

## End-to-end Verification
1. `.venv/bin/python -c "import pandas, sklearn, lightgbm, pytest; print('ok')"` (env).
2. `.venv/bin/pytest tests/test_data_contract.py -q`.
3. `.venv/bin/pytest tests/test_features.py tests/test_validation.py tests/test_submission.py -q`.
4. `.venv/bin/python src/data.py` → prints feature shape, dtypes, and label counts.
5. `.venv/bin/python scripts/01_baseline.py` → prints holdout balanced accuracy + writes
   `submissions/01_baseline.csv` + `experiments/01_baseline.json`.
6. `.venv/bin/python scripts/02_cv_threshold.py` → prints OOF balanced accuracy before/after
   threshold tuning + writes `submissions/02_cv_tuned.csv`.
7. `.venv/bin/python scripts/03_tune.py` → prints repeated-seed OOF score, params, ablations,
   and writes `submissions/03_final.csv`.
8. Optional: `.venv/bin/python scripts/04_ensemble.py` → prints ensemble comparison and writes
   `submissions/04_ensemble.csv`.
9. `wc -l submissions/*.csv` → each = 247436 lines; ids match `sample_submission.csv`.
10. Every score-producing submission is regenerable from its script with no manual steps and
    has a matching experiment ledger entry.
