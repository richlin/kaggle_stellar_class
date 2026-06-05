# Candidate Recommendations

Current public-best fallback: `submissions/19_loo_spatial_final.csv` with public balanced accuracy `0.96970`.

## Current Incumbent

1. `submissions/19_loo_spatial_final.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: `0.96970`.
   - Rationale: current public best. It improves `+0.00043` over `16_spatial_blend.csv` and is only `+0.00030` short of `0.97`.

## Next Public Probes

No immediate public probe is recommended from the existing cached probability files. Revisit with the future plan in `docs/superpowers/plans/2026-06-05-score-over-097-revisit-plan.md`.

## Secondary / Not First

- `submissions/20_loo_spatial_neutral.csv`: public `0.96968`, improved over `16` but trailed `19`.
- `submissions/23_loo_spatial_star_tilt.csv`: public `0.96970`, tied `19` but did not improve; no need to continue STAR tilt without a new signal.
- `submissions/22_loo_spatial_mild_nongal.csv`: public `0.96944`, regressed; do not submit stronger lower-GALAXY variants.
- `submissions/24_loo_spatial_stronger_nongal.csv`: do not submit now; `22` already disproved this direction.
- `submissions/25_loo_spatial_xgb_final.csv`: raw LOO XGBoost blend is too GALAXY-heavy; prefer calibrated `26`.
- `submissions/26_loo_spatial_xgb_calibrated.csv`: public `0.96956`, regressed vs `19`/`23`; stop XGBoost-side LOO probing from this cache.
- `submissions/21_loo_spatial_galaxy_lean.csv`: do not prioritize now; `19` beating `20` suggests the public set did not want more GALAXY.
- `submissions/17_transductive_spatial.csv`: residual blend selected weight `0.0`, identical to `16`; do not submit.
- `submissions/18_galaxy_residual.csv`: threshold search selected no flips, identical to `16`; do not submit.
- `submissions/12_multi_blend.csv`: prior public best `0.96711`, superseded by `16`.

## Next After Public Feedback

- Keep `19`/`23` as the public incumbent pair at `0.96970`.
- Next work should create a new signal or a new validation proxy, not another class-multiplier variant from `19`/`25` cached probabilities.
