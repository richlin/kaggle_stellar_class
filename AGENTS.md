# AGENTS.md

Routing file for AI agents working in this repo. Keep it lean — details live in the linked files.

## What this is

Kaggle competition: **predict the stellar class** (`GALAXY` / `QSO` / `STAR`) for each object.
**Metric: balanced accuracy** (mean per-class recall) — the 65/20/14 class imbalance is the core challenge.

## Start here

- **Plan & task breakdown:** [`tasks/plan.md`](tasks/plan.md) — phased approach, acceptance criteria, verification.
- **Live task list:** [`tasks/todo.md`](tasks/todo.md) — check items off in the same commit as the code change.
- **Data:** `raw_data/{train,test,sample_submission}.csv` (gitignored; download from Kaggle). See plan for schema.

## Conventions

- **Python** managed with `uv`. Activate: `source .venv/bin/activate`. Deps pinned in `requirements.txt`.
- Shared data/feature code in `src/`; one runnable script per phase in `scripts/`; outputs in `submissions/`.
- **Every submission must pass the validator** before it's considered done:
  `python -m pytest tests/ -q` (validates row count, columns `id,class`, label set, id alignment).
- Optimize for **balanced accuracy on out-of-fold predictions**, never on the training fit.

## Deterministic checks (advisory)

- Lint: `ruff check .`
- Tests: `pytest -q`

These are advisory (not enforced via hooks). Run them before committing a phase.
