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
- [x] Task 12: error-margin analysis → low-redshift GALAXY/STAR boundary features (`feature_set="boundary_v1"`)
  - Sprint contract: preserve `baseline` columns exactly; add only deterministic boundary features tied to the known GALAXY/STAR low-redshift leak; accept a new submission only if OOF beats `0.965925` and public-best `03_final` remains the fallback.
  - Verification: feature tests first, `pytest -q`, `ruff check .`, valid submission, experiment JSON with OOF/per-class recall and public-score rationale.
- [x] Task 12 result: `boundary_v1` failed local gate (`0.965630` tuned OOF), so do not submit `submissions/05_boundary_v1.csv`.
- [x] Task 12 follow-up: generate STAR-safe XGBoost blend (`submissions/06_star_safe_blend.csv`) after `04_ensemble` public regression showed STAR-boundary risk.
- [x] Checkpoint E: submitted `submissions/09_extended_seed_average.csv`, `submissions/10_target_encoding.csv`, `submissions/11_target_encoding_blend.csv`, `submissions/12_multi_blend.csv`, and `submissions/13_class_weight_lgbm.csv`; current public best is `12_multi_blend.csv` at `0.96711`.
- [x] Task 13: high-confidence pseudo-labeling generated `submissions/08_pseudolabel.csv`; high risk because no honest OOF and STAR count mirrors failed `04_ensemble` pattern.
- [x] Task 13 alternatives: probability stacker failed local gate (`0.965915`); extended 5-seed average passed local gate (`0.966006`) but is weaker than `06_star_safe_blend`.
- [x] Task 13 target-encoding branch: standalone target encoding failed local gate (`0.965530`), but `submissions/11_target_encoding_blend.csv` improved to `0.966260` OOF and becomes the next candidate.
- [x] Task 17: multi-blend candidate across `03_final`, XGBoost, extended seed averaging, and boundary features; accept only if it beats `11_target_encoding_blend.csv` locally and preserves the public-best fallback rule.
- [x] Task 18: class-adjusted LightGBM candidate using balanced weights nudged toward QSO/STAR; accept only if honest 5-fold OOF or blend beats `12_multi_blend.csv`.

## Phase 6: XGBoost hyperparameter tuning + re-blend
Full detail in [`~/.claude/plans/great-does-hyperparameter-tuning-lexical-diffie.md`]. Rationale: XGBoost is the only non-LightGBM model carrying blend weight (0.30) yet the least tuned (one hand-picked config, fixed `n_estimators=1200`, **no early stopping**). LightGBM was already screened in Phase 4, so tuning effort concentrates on XGBoost.
**Heightened caution:** `04_ensemble` regressed on the public board despite a higher OOF — OOF gains are not transferring cleanly, so leakage control and the public-fallback rule are mandatory, not optional.
- [x] Pre-work: add `optuna` to `requirements.txt`; `VIRTUAL_ENV=.venv uv pip install optuna`
- [x] Task 14: `scripts/05_tune_xgb.py` — Optuna TPE over XGBoost search space; **add early stopping** (rounds=50 on `mlogloss`, `n_estimators` cap ↑); reuse `_xgboost_frames`, `compute_sample_weight("balanced")`, `search_stable_multipliers`
- [ ] Task 14 anti-leakage: **select** params on 3-fold seed=42 proxy; **re-validate** the single winner on full 5-fold × fresh seeds {52,53,54} before its OOF enters any blend
- [x] Task 15: `tests/test_tune_xgb.py` first — objective returns finite float, sampled params within bounds, OOF/test shapes `(577347,3)`/`(247436,3)`, submission preserves id order
- [x] Task 16: smoke re-blend tuned XGBoost with reference LightGBM (reuse `04_ensemble` blend + continuous-multiplier helpers via `importlib`); write `submissions/05_tuned_ensemble.csv` + `experiments/05_tune_xgb.json`
- [x] Task 16 result: 8-trial tuned-XGBoost blend reached `0.966062` OOF, below later local/public candidates.
- [ ] **Checkpoint F:** adopt only if re-blended OOF beats `0.966055` within recall/fold guardrails AND does not repeat the STAR-boundary regression pattern; otherwise record the negative result and keep the public-best fallback. Review with user before any Kaggle upload.

## Phase 7 (last resort)
- [x] Task 13: high-confidence pseudo-labeling

## Phase 8: Spatial neighbourhood features (THE gap to 0.97)
Leaderboard proves 0.97+ is real (top 0.97127). Found: class is clustered in sky position — position-only LGBM = 0.678 bal-acc, neighbours share class 68% vs 49% chance. OOF spatial k-NN class-fraction features lifted a single split 0.9655→0.96828 (+0.0028). Full detail: [`~/.claude/plans/great-does-hyperparameter-tuning-lexical-diffie.md`].
- [x] Task 19: pre-check — test spatially intermixed with train (nearest-train-neighbour distance dist identical at p50/p90/p99). Signal transfers.
- [x] Task 20: `tests/test_spatial_features.py` — unit-sphere conversion, RA wraparound, shapes/finiteness, KFold-OOF leakage audit (flip own label → own feature unchanged). 4 tests pass.
- [x] Task 21: `src/spatial.py` + `scripts/15_spatial_features.py` — unit-sphere (x,y,z); OOF k-NN class fractions (k∈{5,10,25,50,100,250}, uniform+distance-weighted) + nearest-class-distance + density; KFold-OOF for train, full-train→test; prior smoothing.
- [x] Task 22: 3-seed 5-fold LightGBM on baseline+spatial → `submissions/15_spatial.csv`, tuned OOF **0.968894** (+0.0026 over 0.966282).
- [x] Task 23: spatial-aware XGBoost (`scripts/16_spatial_xgb.py`) + blend → `submissions/16_spatial_blend.csv`, tuned OOF **0.969071** (best). Redshift-augmented neighbours rejected (hurt).
- [x] **Checkpoint G:** OOF 0.969071 >> 0.9663; leakage audit passes; `pytest -q` (67), `ruff check .`, `src.validate` all green.
- [x] **Checkpoint H (real gate):** `submissions/16_spatial_blend.csv` public score `0.96927`; accepted as new public incumbent over `12_multi_blend` (`0.96711`).
- [ ] Task 24 (toward leader cluster ~0.9708): GALAXY recall (0.958) is the wall — try finer/position-cell target encoding; remaining public gap is ~0.00073 to `>0.97` and ~0.0015 to the `~0.9708` leader cluster.
