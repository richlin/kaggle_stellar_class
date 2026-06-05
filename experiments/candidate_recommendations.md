# Candidate Recommendations

Current public-best fallback: `submissions/16_spatial_blend.csv` with public balanced accuracy `0.96927`.

## Current Incumbent

1. `submissions/16_spatial_blend.csv`
   - Local OOF: `0.9690706512708674`.
   - Public LB: `0.96927`.
   - Rationale: strongest verified local/public candidate. Keep as fallback until a public submission beats it.

## Next Public Probes

2. `submissions/20_loo_spatial_neutral.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: n/a.
   - Rationale: 441 rows changed vs `16`, GALAXY count `+38`; lowest-risk probe of whether leave-one-out train spatial features transfer better to the public test set.

3. `submissions/21_loo_spatial_galaxy_lean.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: n/a.
   - Rationale: 890 rows changed vs `16`, GALAXY count `+835`; use only after `20` if public feedback suggests GALAXY-leaning movement helps.

## Secondary / Not First

- `submissions/19_loo_spatial_final.csv`: final-only LOO candidate, but it reduces GALAXY count by 192 vs `16`; prefer `20`.
- `submissions/17_transductive_spatial.csv`: residual blend selected weight `0.0`, identical to `16`; do not submit.
- `submissions/18_galaxy_residual.csv`: threshold search selected no flips, identical to `16`; do not submit.
- `submissions/12_multi_blend.csv`: prior public best `0.96711`, superseded by `16`.

## Next After Public Feedback

- If `20` improves, update `experiments/leaderboard.md`, then submit `21` only if the result still implies GALAXY recall is short.
- If both `20` and `21` regress, return to `16` as final and stop local residual search until a new spatial signal is identified.
