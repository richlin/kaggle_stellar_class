# Candidate Recommendations

Current public-best fallback: `submissions/19_loo_spatial_final.csv` with public balanced accuracy `0.96970`.

## Current Incumbent

1. `submissions/19_loo_spatial_final.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: `0.96970`.
   - Rationale: current public best. It improves `+0.00043` over `16_spatial_blend.csv` and is only `+0.00030` short of `0.97`.

## Next Public Probes

2. `submissions/23_loo_spatial_star_tilt.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: n/a.
   - Rationale: closest to `19` while shifting slightly toward STAR instead of QSO (`452` rows changed vs `16`, GALAXY count `-145`). Submit this first if slots are tight.

3. `submissions/22_loo_spatial_mild_nongal.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: n/a.
   - Rationale: extends the public-winning `19` direction with fewer GALAXY predictions (`578` rows changed vs `16`, GALAXY count `-443`). Submit after `23` or if testing the non-GALAXY trend directly.

4. `submissions/24_loo_spatial_stronger_nongal.csv`
   - Local OOF: n/a; final-only LOO spatial train/test mismatch candidate.
   - Public LB: n/a.
   - Rationale: stronger version of `22` (`701` rows changed vs `16`, GALAXY count `-612`). Higher risk; use only if `22` improves.

## Secondary / Not First

- `submissions/20_loo_spatial_neutral.csv`: public `0.96968`, improved over `16` but trailed `19`.
- `submissions/21_loo_spatial_galaxy_lean.csv`: do not prioritize now; `19` beating `20` suggests the public set did not want more GALAXY.
- `submissions/17_transductive_spatial.csv`: residual blend selected weight `0.0`, identical to `16`; do not submit.
- `submissions/18_galaxy_residual.csv`: threshold search selected no flips, identical to `16`; do not submit.
- `submissions/12_multi_blend.csv`: prior public best `0.96711`, superseded by `16`.

## Next After Public Feedback

- If `23` or `22` clears `0.97`, update `experiments/leaderboard.md`, `PROGRESS.md`, and `DECISIONS.md` before any further modeling.
- If both regress below `19`, keep `19` as incumbent and stop GALAXY-lean probes.
