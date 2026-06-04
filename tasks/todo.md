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
- [ ] **Checkpoint B:** tuned OOF beats baseline, submission valid, review with user

## Phase 4: Hyperparameter tuning + polish
- [ ] Task 5: `scripts/03_tune.py` — tune LightGBM params on CV, re-tune thresholds, feature importance review → `submissions/03_final.csv`
- [ ] **Checkpoint C:** final OOF documented, `PROGRESS.md` + `DECISIONS.md` updated, ready for review
