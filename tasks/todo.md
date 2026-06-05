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
- [x] Task 24 (toward leader cluster ~0.9708): GALAXY recall (0.958) is the wall — try graph/cluster target encodings, GALAXY residual correction, and LOO final spatial training.

## Phase 9: Task 24 transductive spatial graph + GALAXY residual correction
Full detail in [`docs/superpowers/plans/2026-06-04-transductive-spatial-task24.md`](../docs/superpowers/plans/2026-06-04-transductive-spatial-task24.md).
- [x] Task 24a: tests first for transductive graph probability features, OOF cluster class rates, meta features, LOO spatial features, and submission id order.
- [x] Task 24b: implement `src/transductive_spatial.py` reusable feature builders with leakage controls.
- [x] Task 24c: implement `scripts/17_transductive_spatial.py` residual LightGBM calibrator and blend search.
- [x] Task 24d: run candidate gates and record results:
  - `17_transductive_spatial.csv`: residual OOF `0.968377`; blend search assigned residual weight `0.0`, so it is identical to `16`.
  - `18_galaxy_residual.csv`: binary GALAXY residual correction selected no flips; OOF unchanged at `0.969071`.
  - `19_loo_spatial_final.csv`: final-only LOO spatial train/test mismatch candidate; no honest OOF, changed 461 rows vs `16` and reduced GALAXY count, so keep as secondary.
  - `20_loo_spatial_neutral.csv`: LOO final variant, 441 rows changed vs `16`, GALAXY count `+38`; first public-risk probe if submission slots are available.
  - `21_loo_spatial_galaxy_lean.csv`: LOO final variant, 890 rows changed vs `16`, GALAXY count `+835`; higher-upside/higher-risk public probe.

## Phase 11: Logit stacking (key learnings from TabPFN-3 public notebook, 2026-06-04)
Source: https://www.kaggle.com/code/philippsinger/tabpfn-3-stacker (reached 0.97+ with TabPFN as meta-learner).
Core principle: convert all OOF probabilities to logits before any blending or meta-learning — logits handle near-certainty predictions geometrically rather than compressing them into [0,1].
**Constraint:** use only non-neural base models and a non-neural meta-learner (LightGBM or logistic regression). Do not use TabPFN or any NN.

- [x] Task 29: logit-blend drop-in — implemented `scripts/29_logit_blend.py`. Result: tied arithmetic blend at OOF=0.969071. FAILED gate (no improvement).
- [x] Task 30: meta-stacking layer — implemented `scripts/30_meta_stacker.py` (2-model: spatial LGBM + spatial XGBoost). Result: OOF=0.968716, FAILED gate (−0.000355 vs incumbent). Root cause: 2 highly correlated spatial base models provide insufficient diversity for meta-learner to add lift.
- [x] Task 31: extended spatial LGBM from 3 to 5 seeds (scripts/32_spatial_5seed_blend.py); running. Also implemented 3-model meta-stacker (scripts/34_3model_meta_stacker.py, waits for script 28) and photometric XGBoost blend (scripts/31_phot_xgb_blend.py, waits for script 28).
- [x] **Checkpoint I:** OOF summary: logit blend FAILED (tied 0.969071), meta-stacker FAILED (0.968716), 5-seed blend PASSED (0.969154 new best). Current best candidate: `submissions/32_spatial_5seed_blend.csv`. `pytest -q` (81 pass), `ruff check .` clean, `src.validate` passes.

## Phase 13: Photometric neighbourhood features (2026-06-05)
- [x] Task P1: `tests/test_photometric_neighbours.py` — 4 leakage/shape/finiteness tests pass.
- [x] Task P2: `scripts/28_photometric_neighbours.py` — DONE. Full 5×3-fold result: tuned OOF **0.968931 (FAILED gate, −0.000140)**. Photometric k-NN redundant with individual color features.
  - Root cause: colour → class is already a STRONG individual predictor; neighbourhood aggregation adds no new information (unlike spatial position which is a WEAK individual predictor).
- [x] Task P3: `scripts/31_phot_xgb_blend.py` — implemented but DEPRIORITIZED (photometric features hurt OOF; adding XGBoost won't fix this).
- [x] Task P4: `scripts/32_spatial_5seed_blend.py` — DONE. OOF **0.969154 (PASSED, +0.000083)**. New best honest OOF. Submission generated.
- [x] Task P5: `scripts/33_loo_family.py` — DONE. "Shallower" variant chosen (in-band: GALAXY 156566, QSO 51275, STAR 39594); final-only candidate. Submission generated.
- [x] Task P6: `scripts/34_3model_meta_stacker.py` — DEPRIORITIZED (photometric features hurt, so 3-model meta-stacker with photometric won't help).
- [x] Task P7: `scripts/35_loo_phot_final.py` — DEPRIORITIZED (photometric k-NN redundant).
- [x] Task P8: `scripts/36_spatial_more_trees.py` — 1500-tree spatial LGBM. Running. Both single-fold models hit max trees at n_est=900 → capacity constraint.
  - Also implemented: `scripts/38_spatial_5seed_1500trees.py` (chains after 36), `scripts/39_spatial_lower_lr.py` (lr=0.02, n_est=3000).
- [x] Task P9: `experiments/next_submission_report.md` — created. Updates pending script 36 result.

## Phase 14: Public feedback consolidation + original dataset append
Sprint contract: update tracking files with public scores before running new experiments. Treat `32_spatial_5seed_blend.csv` as public incumbent (`0.96977`). Treat `41_5seed_lgbm_xgb_catboost.csv` as an OOF/public mismatch until proven otherwise. Do not submit more CatBoost-blend or multiplier-only variants without new validation evidence.
- [x] Task 43: record public scores from Kaggle:
  - `32_spatial_5seed_blend.csv`: public `0.96977`, current public best.
  - `41_5seed_lgbm_xgb_catboost.csv`: public `0.96958`, regressed vs `32` despite best honest OOF.
- [x] Task 44: verify competition-discussion categorical formulae on local train+test:
  - `spectral_type = pd.cut(r-g, [-inf,-1,-0.5,0,inf], labels=["M","G/K","A/F","O/B"])`.
  - `galaxy_population = pd.cut(u-r, [-inf,2.2,inf], labels=["Blue_Cloud","Red_Sequence"])`.
  - Result: both matched all `824,782` combined train/test rows.
- [x] Task 45: add a deterministic test for the categorical formulae so future agents do not treat these columns as mysterious external signal.
- [ ] Task 46: locate and stage the original labelled dataset outside `raw_data/` or under a clearly named ignored path; document source, row count, columns, class mapping, and licence/competition acceptability before modeling.
- [x] Task 47: build `scripts/43_original_append_audit.py` to recreate categoricals for original rows, align schema to competition train, check duplicate ids/features against train/test, check class distribution and feature shift, and write `experiments/43_original_append_audit.json`.
- [ ] Task 48: only if Task 47 passes, train an appended-data candidate with validation scored solely on competition train OOF folds; accept only if OOF beats `0.969202` and class recalls do not regress, then compare public-risk diagnostics against `32`.
  - Script scaffold exists in `scripts/44_original_append_train.py`, but execution is blocked until Task 46 locates a dataset and Task 47 produces an audit PASS for the same file path.
- [x] Task 51: harden original-append scaffolding before any dataset run:
  - Sprint contract: treat Task 46 as blocked until the original dataset is staged; do not train or submit without an audit PASS.
  - Add tests for categorical formula mismatch detection, duplicate feature overlap detection, class-column validation, and audit/train path consistency.
  - Patch only the audit/train guardrails needed for those tests.
  - Verification: targeted tests, `uv run pytest -q`, `uv run ruff check .`, `git diff --check`.

## Phase 15: Plan consolidation
- [x] Task 49: replace stale bootstrap-era `tasks/plan.md` with the current operating plan.
- [x] Task 50: update `AGENTS.md` start-here routing so future agents read `PROGRESS.md`, `tasks/plan.md`, `tasks/todo.md`, `experiments/leaderboard.md`, and `DECISIONS.md` in the right order.

## Phase 12: Discovery EDA Notebook
Full detail in [`docs/superpowers/plans/2026-06-05-eda-discovery-notebook.md`](../docs/superpowers/plans/2026-06-05-eda-discovery-notebook.md).
- [x] Task 32: create `notebooks/eda_discovery.ipynb` for score-discovery EDA: schema, class balance, numeric distributions, redshift ambiguity, color/magnitude views, categorical signal, train/test shift, spatial structure, residual hooks, and next hypotheses.
- [x] Task 33: add notebook generation and structure checks via `scripts/create_eda_discovery_notebook.py` and `tests/test_eda_notebook.py`.
- [x] Checkpoint J: execute all notebook code cells locally; run `uv run pytest -q`; run `uv run ruff check .`.
