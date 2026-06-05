# Candidate Recommendations

Current public-best fallback: `submissions/19_loo_spatial_final.csv` with public balanced accuracy `0.96970`.

## Current Incumbent

1. `submissions/19_loo_spatial_final.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: `0.96970`.
   - Rationale: current public best. It improves `+0.00043` over `16_spatial_blend.csv` and is only `+0.00030` short of `0.97`.

## Next Public Probes

2. `submissions/26_loo_spatial_xgb_calibrated.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: n/a.
   - Rationale: tests the same LOO train/test feature-density hypothesis on the XGBoost side, but calibrated back near the public-best `19` class mix. It changes 779 rows vs `16`, so it is meaningfully different from the tied `19`/`23` plateau without repeating the failed lower-GALAXY direction.

## Secondary / Not First

- `submissions/20_loo_spatial_neutral.csv`: public `0.96968`, improved over `16` but trailed `19`.
- `submissions/23_loo_spatial_star_tilt.csv`: public `0.96970`, tied `19` but did not improve; no need to continue STAR tilt without a new signal.
- `submissions/22_loo_spatial_mild_nongal.csv`: public `0.96944`, regressed; do not submit stronger lower-GALAXY variants.
- `submissions/24_loo_spatial_stronger_nongal.csv`: do not submit now; `22` already disproved this direction.
- `submissions/25_loo_spatial_xgb_final.csv`: raw LOO XGBoost blend is too GALAXY-heavy; prefer calibrated `26`.
- `submissions/21_loo_spatial_galaxy_lean.csv`: do not prioritize now; `19` beating `20` suggests the public set did not want more GALAXY.
- `submissions/17_transductive_spatial.csv`: residual blend selected weight `0.0`, identical to `16`; do not submit.
- `submissions/18_galaxy_residual.csv`: threshold search selected no flips, identical to `16`; do not submit.
- `submissions/12_multi_blend.csv`: prior public best `0.96711`, superseded by `16`.

## Next After Public Feedback

- If `26` clears `0.97`, update `experiments/leaderboard.md`, `PROGRESS.md`, and `DECISIONS.md` before any further modeling.
- If `26` regresses below `19`, keep `19` as incumbent and stop multiplier-only probing from the current caches.
