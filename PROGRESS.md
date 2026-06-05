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
- 2026-06-04: `submissions/03_final.csv` official public score: `0.96691`.
  - New target: exceed `0.97` public balanced accuracy.
  - Required public lift: approximately `+0.0031`.
  - Improvement plan saved in `docs/superpowers/plans/2026-06-04-score-over-097-improvement.md`.
- 2026-06-04: Implemented Phase 5 cross-library ensemble in `scripts/04_ensemble.py`.
  - Added tested helpers for weighted probability blending, blend-weight search, continuous threshold tuning, and submission generation.
  - Trained 5-fold XGBoost, CatBoost, and LightGBM DART candidates on the same fold protocol as the reference model.
  - Best local blend: `lgbm_seed_average_final=0.7`, `xgboost=0.3`, `catboost=0.0`, `lgbm_dart=0.0`.
  - Final Phase 5 tuned OOF balanced accuracy: `0.9660551711148327`.
  - Chosen Phase 5 multipliers: `[0.750997062181838, 0.7959641054598435, 0.8981824050203887]`.
  - Generated `submissions/04_ensemble.csv`, `experiments/04_ensemble.json`, and candidate OOF/test probability arrays.
- 2026-06-04: `submissions/04_ensemble.csv` official public score: `0.96676`.
  - This regressed from `03_final.csv` public score `0.96691` despite a small local OOF lift.
  - Diagnosis: `04_ensemble` changed only 700 test rows, mostly moving prior `STAR` predictions to `GALAXY`/`QSO`; public score suggests that STAR-boundary trade did not transfer.
- 2026-06-04: Implemented and tested `feature_set="boundary_v1"` plus `scripts/05_boundary_features.py`.
  - Boundary feature candidate tuned OOF balanced accuracy: `0.9656299274771866`.
  - This failed the local gate versus `03_final` OOF `0.9659249816190973`, so `submissions/05_boundary_v1.csv` is not a recommended public submission.
- 2026-06-04: Generated STAR-safe blend candidate in `scripts/06_star_safe_blend.py`.
  - Weights: `lgbm_seed_average_final=0.6`, `xgboost=0.4`.
  - Multipliers: `[0.8, 1.1, 1.15]`.
  - OOF balanced accuracy: `0.9662339484478014` (`+0.00030896682870407144` vs `03_final`).
  - Recall deltas vs `03_final`: `GALAXY=-0.002378933983257392`, `QSO=+0.003414629982158579`, `STAR=-0.00010879551278952793`.
  - Generated `submissions/06_star_safe_blend.csv` and `experiments/06_star_safe_blend.json`.
- 2026-06-04: Tested additional score-push candidates after `06_star_safe_blend.csv`.
  - `scripts/07_probability_stacker.py`: logistic stacker over saved OOF probabilities; tuned OOF `0.965914781047898`, failed local gate vs `03_final`.
  - Train/test exact and rounded duplicate check: no matches through 2 decimal places, so no duplicate-label leakage opportunity.
  - Unweighted LightGBM: tuned OOF `0.9594214999106119`; high GALAXY recall but STAR recall collapsed, not useful.
  - Square-root-balanced LightGBM: tuned OOF `0.9635880281238342`; failed local gate.
  - `scripts/08_pseudolabel.py`: generated low-weight high-confidence pseudo-label candidate with 150,737 pseudo rows, but no honest OOF score and STAR count risk resembles failed `04_ensemble`.
  - `scripts/09_extended_seed_average.py`: extended public-best LightGBM average from 3 to 5 seeds; tuned OOF `0.966006054665383`, valid but weaker than `06_star_safe_blend`.
- 2026-06-04: Started Phase 6 XGBoost tuning setup.
  - Installed and pinned `optuna==4.9.0` and dependencies.
  - Confirmed XGBoost 3.2 early stopping must be passed through `XGBClassifier(..., early_stopping_rounds=...)`, not `fit()`.
- 2026-06-04: Implemented and smoke-tested `scripts/05_tune_xgb.py`.
  - Added `tests/test_tune_xgb.py`; helper tests cover parameter bounds, one-hot encoding, tiny objective scoring, and submission id order.
  - 8-trial Optuna screen best score: `0.9649650390017835`.
  - Tuned XGBoost seed-52 argmax OOF: `0.9653085627646719`.
  - Final tuned-XGBoost blend OOF: `0.966062166950432`, below `06_star_safe_blend` OOF `0.9662339484478014`.
  - Generated `submissions/05_tuned_ensemble.csv`, but it is not the recommended next submission because it is weaker than `06` locally and moves more STAR rows than `06`.
- 2026-06-04: Added `experiments/candidate_recommendations.md` to rank submission candidates and avoid weaker artifacts.
- 2026-06-04: Implemented leakage-safe target encoding experiment and blend.
  - `scripts/10_target_encoding.py`: OOF-safe target encodings for `spectral_type`, `galaxy_population`, `spectral_population`, `redshift_bin`, and `spectral_population_redshift_bin`.
  - Standalone target-encoding LightGBM tuned OOF: `0.9655299670953859`, failed local gate.
  - `scripts/11_target_encoding_blend.py`: blend `03_final=0.5`, `xgboost=0.4`, `target_encoding=0.1` with multipliers `[0.8, 1.1, 1.15]`.
  - Target-encoding blend OOF: `0.9662595764879448`, now the strongest local safe candidate.
  - STAR recall delta vs `03_final`: `-0.00007253034185961127`.
  - Generated `submissions/11_target_encoding_blend.csv` and `experiments/11_target_encoding_blend.json`.
- 2026-06-04: Implemented cached multi-blend candidate in `scripts/12_multi_blend.py`.
  - Blend: `03_final=0.23`, `xgboost=0.44`, `extended_seed_average=0.28`, `boundary_v1=0.05`.
  - Multipliers: `[0.74, 0.94, 1.05]`.
  - Multi-blend OOF: `0.9662824834818386`, now the strongest local candidate.
  - Recall deltas vs `03_final`: `GALAXY=-0.0014543816890960626`, `QSO=+0.0026719479610390895`, `STAR=-0.00014506068371933356`.
  - Generated `submissions/12_multi_blend.csv`, `experiments/12_multi_blend.json`, and cached `12_multi_blend_*_probabilities.npy`.
  - Transition check vs `03_final`: 793 test rows changed; class counts `GALAXY=156524`, `QSO=51329`, `STAR=39582`.
- 2026-06-04: Recorded new public leaderboard results from Kaggle submissions.
  - `09_extended_seed_average.csv`: public `0.96658`.
  - `10_target_encoding.csv`: public `0.96673`.
  - `11_target_encoding_blend.csv`: public `0.96700`.
  - `12_multi_blend.csv`: public `0.96711`, current public best.
  - `13_class_weight_lgbm.csv`: public `0.96692`.
  - Remaining lift required to exceed `0.97`: about `+0.00289`.
- 2026-06-04: Class-adjusted LightGBM did not supersede `12_multi_blend`.
  - Standalone OOF: `0.9658524584206448`.
  - Blend OOF: `0.9661492957452017` (`-0.00013318773663695271` vs `12_multi_blend`).
  - Public score: `0.96692`, below `12_multi_blend` public `0.96711`.
  - Keep as a recorded negative result, not a next submission path.

- 2026-06-04: **BREAKTHROUGH — spatial neighbourhood features (Phase 8). The earlier "0.966 feature ceiling" conclusion was wrong** (disproved by the leaderboard: top `0.97127`, cluster `~0.9708`).
  - Found via read-only EDA: class is clustered in sky position. Position-only LightGBM = `0.678` balanced accuracy; 10 nearest objects by `(alpha,delta)` share a class `0.68` of the time vs `0.49` chance. No leak (zero id/feature overlap), no train/test shift.
  - Pre-check: test points are spatially intermixed with train (matched nearest-train-neighbour distance distributions), so spatial features transfer.
  - `src/spatial.py` + `scripts/15_spatial_features.py`: leakage-safe out-of-fold k-NN class-fraction features on unit-sphere `(alpha,delta)` (k∈{5,10,25,50,100,250}, uniform + distance-weighted, nearest-class distance, density; KFold-OOF for train, full-train→test, prior smoothing). `tests/test_spatial_features.py` (4 tests) includes a leakage audit.
  - Spatial LightGBM tuned OOF `0.968894` (`submissions/15_spatial.csv`); STAR recall `0.952→0.975`.
  - `scripts/16_spatial_xgb.py`: spatial-aware XGBoost + blend → tuned OOF **`0.969071`** (`submissions/16_spatial_blend.csv`, `0.55*spatial_lgbm + 0.45*spatial_xgb`), recalls GALAXY `0.958` / QSO `0.975` / STAR `0.974`. Best local candidate.
  - Rejected: redshift-augmented neighbours (hurt: `0.9665` vs `0.9686`); greedy blend of spatial + non-spatial models (non-spatial models too weak).
  - All gates green: `pytest -q` (67), `ruff check .`, `src.validate`.
- 2026-06-04: `submissions/16_spatial_blend.csv` official public score: `0.96927`.
  - New public incumbent over `12_multi_blend.csv` (`0.96711`) by `+0.00216`.
  - Remaining lift required to exceed `0.97`: about `+0.00073`.

## In Progress

- None.

## Blockers

- None.

## Next Steps

- Since `16_spatial_blend.csv` is still short of the target `>0.97`, continue with Task 24: GALAXY recall (`0.958`) is the binding constraint — try finer / position-cell target encoding for the remaining gap.
