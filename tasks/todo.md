# TODO — Predicting Stellar Class

Check items off in the **same commit** as the code change. Full detail in [`plan.md`](plan.md).

## Phase 0: Setup
- [x] git init + `.gitignore`
- [x] `uv` venv + pinned `requirements.txt` (pandas, scikit-learn, lightgbm, numpy, pytest, ruff)
- [x] Advisory checks: `ruff.toml` + submission validator (`src/validate.py`) + smoke test (`tests/test_validate.py`)
- [x] Data-contract checks for raw Kaggle CSVs (`tests/test_data_contract.py`)
- [x] `AGENTS.md` routing file
- [x] APM init + workflow skills

## Phase 1: End-to-end baseline (first scoreable submission)
- [x] Task 1: `src/data.py` — load + features (color indices, categorical dtype), label encoding
- [x] Task 2: `scripts/01_baseline.py` — stratified holdout, class-weighted LightGBM, write `submissions/01_baseline.csv`
- [x] **Checkpoint A:** submission valid (`pytest -q`), baseline balanced accuracy recorded, review with user

## Phase 2: Cross-validated training
- [x] Task 3: `scripts/02_cv_threshold.py` (CV part) — 5-fold OOF probabilities + bagged test probs

## Phase 3: Threshold / prior tuning
- [x] Task 4: extend `scripts/02_cv_threshold.py` — tune per-class multipliers on OOF to maximize balanced accuracy → `submissions/02_cv_tuned.csv`
- [x] **Checkpoint B:** tuned OOF beats baseline, submission valid, review with user

## Phase 4: Hyperparameter tuning + polish
- [x] Task 5: `scripts/03_tune.py` — tune LightGBM params on CV, re-tune thresholds, feature importance review → `submissions/03_final.csv`
- [x] **Checkpoint C:** final OOF documented, `PROGRESS.md` + `DECISIONS.md` updated, ready for review

## Phase 5: Cross-library ensemble + continuous thresholds (target public > 0.97)
Full detail in [`docs/superpowers/plans/2026-06-04-score-over-097-improvement.md`](../docs/superpowers/plans/2026-06-04-score-over-097-improvement.md) (revised 2026-06-04 with 4 modifications). Worktree: `phase5-ensemble`.
- [x] Pre-work: OOF confusion analysis — leak is GALAXY→STAR (11,497) + GALAXY→QSO (4,747), a low-redshift boundary (recorded in plan)
- [x] Pre-work: install + verify `xgboost`, `catboost`, `scipy` in `.venv`
- [x] Task 6: record official `0.96691` in `experiments/leaderboard.md` + `PROGRESS.md`
- [x] Task 7: `tests/test_ensemble.py` (RED) — weighted blend, blend-weight search, continuous threshold search, submission id-order
- [x] Task 8: `scripts/04_ensemble.py` — blend helpers + `search_continuous_multipliers` (Nelder-Mead, stability-guarded) → GREEN
- [x] Task 9: train **XGBoost** (primary diversity) + **`lgbm_dart`** on identical 5-fold splits; save OOF/test probs
- [x] Task 10: train **CatBoost** (native categoricals); graceful fallback if install/runtime fails
- [x] Task 11: grid-search blend weights across families, then continuous threshold tuning on blended OOF → `submissions/04_ensemble.csv`
- [x] **Checkpoint D:** OOF beats `0.965925`, recalls stay within guardrails, submission valid (`pytest -q`, `ruff check .`, `src.validate`), review with user before Kaggle submit
- [ ] Task 12 (only if ensemble < 0.97): error-margin analysis → low-redshift GALAXY/STAR boundary features (`feature_set="boundary_v1"`)
- [ ] Task 13 (last resort): high-confidence pseudo-labeling
