# Progress

## Completed

- 2026-06-04: Added raw-data contract coverage for train/test/sample CSV shape, ids, labels, missing values, and categorical level coverage.
- 2026-06-04: Implemented Phase 1 shared data/feature layer:
  - `src/data.py` loads raw CSVs, builds features, and label-encodes `GALAXY` / `QSO` / `STAR`.
  - `src/features.py` builds baseline numeric, color, magnitude-summary, coordinate, redshift-interaction, and categorical-interaction features.
- 2026-06-04: Implemented `scripts/01_baseline.py`:
  - Stratified 80/20 holdout.
  - Class-weighted LightGBM multiclass baseline.
  - Generated `submissions/01_baseline.csv` (gitignored).
  - Wrote `experiments/01_baseline.json`.
- 2026-06-04: Phase 1 baseline holdout balanced accuracy: `0.964832757657062`.
  - Per-class recall: `GALAXY=0.9554943308254742`, `QSO=0.9738785266123181`, `STAR=0.9651254155333938`.

## In Progress

- Checkpoint A: user review of the valid Phase 1 baseline before starting Phase 2 cross-validation.

## Blockers

- None.

## Next Steps

- Review Phase 1 baseline results and optionally submit `submissions/01_baseline.csv` to Kaggle.
- After Checkpoint A review, implement Phase 2: 5-fold OOF LightGBM probability harness in `scripts/02_cv_threshold.py`.
