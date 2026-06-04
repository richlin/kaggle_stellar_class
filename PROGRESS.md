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
- 2026-06-04: Implemented Phase 2/3 CV and threshold harness in `scripts/02_cv_threshold.py`.
  - 5-fold OOF argmax balanced accuracy: `0.9649380171780901`.
  - Unconstrained tuned candidate balanced accuracy: `0.965167921449232`.
  - Stable chosen tuned balanced accuracy: `0.9651028582582045`.
  - Chosen multipliers: `[0.75, 0.75, 0.9]`.
  - Generated `submissions/02_cv_tuned.csv` (gitignored), `experiments/02_cv_threshold.json`, and local probability arrays.

## In Progress

- Checkpoint B: user review of the valid Phase 2/3 tuned CV submission before starting Phase 4 hyperparameter tuning.

## Blockers

- None.

## Next Steps

- Review Phase 2/3 tuned CV results and optionally submit `submissions/02_cv_tuned.csv` to Kaggle.
- After Checkpoint B review, implement Phase 4: hyperparameter tuning and feature-family ablations in `scripts/03_tune.py`.
