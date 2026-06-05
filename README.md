# Predicting Stellar Class

Kaggle competition: classify each object as `GALAXY`, `QSO`, or `STAR` from SDSS photometric and
spectroscopic features.

**Metric:** balanced accuracy (mean per-class recall). The 65 / 20 / 14 class split means a
majority-class-biased model scores poorly — handling imbalance *is* the problem.

**Current public best:** `0.96977` (`submissions/32_spatial_5seed_blend.csv`).  
**Target:** > 0.97. Remaining gap: ~+0.00023.

---

## Key results

| Submission | OOF | Public | Notes |
|---|---:|---:|---|
| `03_final.csv` | 0.9659 | 0.96691 | Phase 4 baseline; 3-seed LGBM + multiplier tuning |
| `12_multi_blend.csv` | 0.9663 | 0.96711 | Best non-spatial blend |
| `16_spatial_blend.csv` | 0.9691 | 0.96927 | Breakthrough: spatial k-NN features |
| `19_loo_spatial_final.csv` | — | 0.96970 | LOO spatial (no honest OOF) |
| **`32_spatial_5seed_blend.csv`** | **0.9692** | **0.96977** | **Current public best** |

Full history: `experiments/leaderboard.md`

---

## Setup

Python is managed with **`uv`**; the venv is pip-less.

```bash
# Install dependencies
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# Run scripts (src.* is bootstrapped via sys.path in each script)
uv run python scripts/03_tune.py

# Tests (also the submission gate)
pytest -q

# Lint (advisory, not enforced)
ruff check .

# Validate a submission before uploading
python -m src.validate submissions/<name>.csv
```

`raw_data/`, `.venv/`, `submissions/`, and `experiments/*.npy` are gitignored — they exist only in
the main checkout. In a fresh worktree, symlink them back.

---

## Architecture

Four pieces explain the whole pipeline:

| File | Role |
|---|---|
| `src/features.py` | Pure feature math — colors, magnitude summaries, coordinate sin/cos, redshift interactions, categorical-interaction string. No I/O. |
| `src/spatial.py` | OOF-safe k-NN class-fraction features on unit-sphere `(alpha, delta)`, k ∈ {5, 10, 25, 50, 100, 250}. The breakthrough feature family. |
| `src/data.py` | `load_raw()` + `build_features()` + label encoding with a **fixed** class order `["GALAXY","QSO","STAR"]`. |
| `src/validation.py` | `balanced_accuracy`, `per_class_recall`, threshold-multiplier tuning, experiment loggers. |

### Two load-bearing conventions

1. **Predictions = `argmax(probabilities × per-class multipliers)`**, not raw argmax. Multipliers are
   tuned on OOF probabilities with stability guards (`search_stable_multipliers` rejects vectors that
   regress per-class recall or per-fold score beyond defined thresholds). Always tune on OOF, never
   on the training fit.

2. **Every score-producing run leaves a reproducible trail**: OOF `.npy`, test `.npy`,
   `experiments/NN_*.json`, a row in `experiments/runs.jsonl`, `submissions/NN_*.csv`, and a
   validator call. CV uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`; the final
   model averages probabilities across seeds `[42, 43, 44]` (5-seed variant adds 45, 46).

---

## Key findings

- **Spatial clustering is the strongest signal.** Sky position alone achieves 0.678 balanced
  accuracy; 10 nearest neighbours share a class 68% of the time (vs 49% by chance). Adding spatial
  k-NN features jumped OOF from ~0.966 to ~0.969.
- **Photometric k-NN does not help.** Individual color features are already strong predictors, so
  k-NN aggregation over them adds no independent information.
- **Tree model diversity ceiling is ~0.969–0.970.** All LGBM / XGBoost / CatBoost combinations on
  current features plateau here. The next jump requires new data (original SDSS labels for append)
  or a fundamentally different feature family.
- **Class multiplier calibration matters more than blend weight tuning** at this performance level.

---

## Phase scripts

| Script | Phase | Description |
|---|---|---|
| `01_baseline.py` | 1 | Stratified holdout LightGBM baseline |
| `02_cv_threshold.py` | 2–3 | 5-fold CV + multiplier tuning |
| `03_tune.py` | 4 | Hyperparameter screen + 3-seed average |
| `15_spatial_features.py` | 8 | Spatial k-NN feature generation |
| `16_spatial_xgb.py` | 8 | Spatial LGBM + XGBoost blend |
| `32_spatial_5seed_blend.py` | 13 | 5-seed spatial LGBM + XGBoost (current best) |
| `43_original_append_audit.py` | 14 | Audit external SDSS dataset before append |
| `44_original_append_train.py` | 14 | Append-augmented training (blocked on staged data) |

---

## Tracking

- `PROGRESS.md` — session-by-session work log and blockers
- `DECISIONS.md` — non-obvious architectural choices
- `tasks/todo.md` — live checklist (check items off in the same commit as the code)
- `tasks/plan.md` — full phased plan
- `experiments/leaderboard.md` — OOF vs public score per submission
