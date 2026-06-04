# Leaderboard

| Date | Submission | Local Metric | Local Score | Public LB | Delta | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-06-04 | `submissions/01_baseline.csv` | Holdout balanced accuracy | 0.964833 | n/a | n/a | Phase 1 baseline; not recorded as submitted. |
| 2026-06-04 | `submissions/02_cv_tuned.csv` | Stable tuned OOF balanced accuracy | 0.965103 | n/a | n/a | Phase 2/3 CV threshold candidate; not recorded as submitted. |
| 2026-06-04 | `submissions/03_final.csv` | Repeated-seed tuned OOF balanced accuracy | 0.965925 | 0.96691 | 0.000985 | Phase 4 final candidate; official public score from Kaggle. |
| 2026-06-04 | `submissions/04_ensemble.csv` | Phase 5 ensemble tuned OOF balanced accuracy | 0.966055 | n/a | n/a | Blend of `lgbm_seed_average_final=0.7` and `xgboost=0.3`; pending public submission decision. |
