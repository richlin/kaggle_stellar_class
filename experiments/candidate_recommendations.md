# Candidate Recommendations

Current public-best fallback: `submissions/12_multi_blend.csv` with public balanced accuracy `0.96711`.

## Current Incumbent

1. `submissions/12_multi_blend.csv`
   - Local OOF: `0.9662824834818386`.
   - Public LB: `0.96711`.
   - Weights: `lgbm_seed_average_final=0.23`, `xgboost=0.44`, `extended_seed_average=0.28`, `boundary_v1=0.05`.
   - Rationale: strongest local and public candidate so far.
   - Risk: still short of the target by about `0.00289`.

## Secondary Public Results

2. `submissions/11_target_encoding_blend.csv`
   - Local OOF: `0.9662595764879448`.
   - Public LB: `0.96700`.
   - Rationale: second-best public score, but superseded by `12_multi_blend.csv`.

3. `submissions/13_class_weight_lgbm.csv`
   - Local OOF: `0.9661492957452017`.
   - Public LB: `0.96692`.
   - Rationale: class-adjusted training did not beat `12` locally or publicly; do not continue this line without new evidence.

4. `submissions/10_target_encoding.csv`
   - Local OOF: `0.9655299670953859`.
   - Public LB: `0.96673`.
   - Rationale: standalone target encoding trails the blend variants.

5. `submissions/09_extended_seed_average.csv`
   - Local OOF: `0.966006054665383`.
   - Public LB: `0.96658`.
   - Rationale: lower-risk 5-seed LightGBM average, but weaker publicly than `12`, `11`, `13`, and `10`.

## Do Not Prioritize

- `submissions/04_ensemble.csv`: public score `0.96676`, worse than `03_final`.
- `submissions/05_tuned_ensemble.csv`: local OOF `0.966062166950432`, weaker than `06_star_safe_blend` and moves more STAR rows.
- `submissions/07_probability_stacker.csv`: local OOF `0.965914781047898`, failed local gate.
- `submissions/08_pseudolabel.csv`: no honest OOF score and STAR count resembles the failed `04_ensemble` pattern.
- `submissions/05_boundary_v1.csv`: local OOF `0.9656299274771866`, failed local gate.
- `submissions/10_target_encoding.csv`: local OOF `0.9655299670953859`, failed local gate as a standalone model.

## Next After Public Feedback

- Current public best is `0.96711`, so the remaining lift to exceed `0.97` is about `+0.00289`.
- Continue only with candidates that add a materially new signal or a stronger validation proxy; cached blend and threshold micro-gains are likely saturated.
