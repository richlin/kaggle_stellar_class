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
- 2026-06-04: Implemented Phase 4 tuning/polish in `scripts/03_tune.py`.
  - Screened three manual LightGBM parameter candidates; `phase3_like` remained best on single-seed CV.
  - Ran six feature-family ablations; coordinate features were clearly required, while no feature removal justified changing the final feature set.
  - Repeated the selected candidate over seeds `42`, `43`, and `44`, averaged probabilities, and re-tuned stable class multipliers.
  - Final repeated-seed OOF balanced accuracy: `0.9659249816190973`.
  - Chosen final multipliers: `[0.9, 0.8, 1.15]`.
  - Mean train-vs-validation overfit gap across repeated final runs: `0.0185067713843499`.
  - Generated `submissions/03_final.csv` (gitignored), `experiments/03_tune.json`, `experiments/runs.jsonl`, and local probability arrays.

## In Progress

- Final review before any Kaggle submission or optional Phase 5 ensemble work.

## Blockers

- None.

## Next Steps

- Review `submissions/03_final.csv` and optionally submit to Kaggle.
- Optional Phase 5: model diversity / ensemble only if the final single-model submission needs a high-score push.
