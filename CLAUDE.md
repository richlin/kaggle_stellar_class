# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Kaggle competition: predict the stellar class (`GALAXY` / `QSO` / `STAR`) per object.
**The metric is balanced accuracy (mean per-class recall).** With a 65/20/14 class split,
a model that favors the majority GALAXY class scores poorly — handling the imbalance *is* the
problem. Current best public score: **0.96691** (`submissions/03_final.csv`); active goal is
> 0.97 (see `tasks/todo.md` Phase 5 and `docs/superpowers/plans/2026-06-04-score-over-097-improvement.md`).

## Environment & commands

Python is managed with **`uv`**; the venv at `.venv/` is **pip-less**. Two gotchas that will bite:

```bash
# Install/add deps — `.venv/bin/pip` does NOT exist:
VIRTUAL_ENV=.venv uv pip install <pkg>        # then pin into requirements.txt

# Run anything (scripts import `src.*`; they bootstrap repo root onto sys.path):
uv run python scripts/03_tune.py              # or: .venv/bin/python scripts/03_tune.py
```

```bash
pytest -q                                     # full suite (also the submission gate)
pytest tests/test_tune.py -q                  # one file
pytest tests/test_tune.py::test_name -q       # one test
ruff check .                                  # advisory lint (not enforced by hooks)
python -m src.validate submissions/<x>.csv    # submission-format guard (run before any Kaggle upload)
```

`raw_data/`, `.venv/`, `submissions/`, and `experiments/*.npy` are **gitignored** — they exist only
in the main checkout. A fresh git worktree will not have them; symlink them back if you work in one.

## Architecture

The pipeline is a layered, phase-scripted flow. Understanding these four pieces explains the whole repo:

- **`src/features.py`** — pure, deterministic feature math (`build_feature_frame`). No I/O. Tiny
  fixtures test it. Colors (`u_g`…), magnitude summaries, coordinate sin/cos, redshift interactions,
  and a `spectral_population` categorical-interaction string. Categoricals are returned as pandas
  `category` dtype for LightGBM's native handling.
- **`src/data.py`** — `load_raw()` (CSVs) + `build_features()` (calls the builder and label-encodes
  the target). The label encoder is fit from a **fixed class order `["GALAXY","QSO","STAR"]`**, never
  inferred from the input frame — so class→index mapping is stable across folds, fixtures, and decoding.
- **`scripts/NN_*.py`** — one runnable script per phase (`01_baseline` → `02_cv_threshold` →
  `03_tune`, and Phase 5's planned `04_ensemble`). Each is invoked by file path and bootstraps
  `sys.path` to the repo root itself.
- **`src/validation.py`** — shared metrics + tuning + logging: `balanced_accuracy`, `per_class_recall`,
  the threshold-multiplier machinery, and `write_json` / `append_jsonl` experiment loggers.
  (`src/validate.py` is unrelated — it's the *submission format* validator.)

### Two conventions that are load-bearing

1. **Predictions are `argmax(probabilities × per-class multipliers)`, not raw argmax.** The multipliers
   are reweighted class priors tuned **on out-of-fold (OOF) probabilities** to maximize balanced
   accuracy. This is how the imbalance is corrected at decision time. Tuning is **stability-guarded**:
   `search_stable_multipliers` (in `scripts/03_tune.py`) rejects any multiplier vector that regresses a
   per-class recall beyond `MATERIAL_CLASS_RECALL_REGRESSION` or a per-fold score beyond
   `MATERIAL_FOLD_REGRESSION`, so the OOF gain doesn't come from overfitting one fold. **Always tune on
   OOF, never on the training fit.**

2. **Every score-producing run leaves a reproducible trail.** A phase run saves OOF probabilities
   (`.npy`), test probabilities (`.npy`), a `experiments/NN_*.json` record (OOF score, per-class recall,
   chosen multipliers, submission path), appends a row to `experiments/runs.jsonl`, writes
   `submissions/NN_*.csv`, and calls the validator. CV uses
   `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`; the final model averages
   probabilities across seeds `[42, 43, 44]` on identical splits. **A submission is "done" only after
   `pytest -q` passes and `src.validate` accepts it.**

When adding a new model family (e.g. Phase 5 XGBoost/CatBoost), train it on the **same fold splits**
as the existing models so OOF rows align and probability blending is valid.

## Tracking & process

- `experiments/leaderboard.md` — OOF vs. public score per submission. Update it after each Kaggle result.
- `PROGRESS.md` / `DECISIONS.md` — running log of work and every non-obvious architectural choice.
- `tasks/plan.md` (full phased plan) and `tasks/todo.md` (live checklist) — check items off in the
  same commit as the code change.
