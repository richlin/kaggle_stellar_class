# Next Submission Report — 2026-06-05

**Public incumbent:** `32_spatial_5seed_blend.csv` at **0.96977**  
**Remaining lift to 0.97:** +0.00023  
**Best honest OOF:** `41_5seed_lgbm_xgb_catboost.csv` at **0.969202**, but it scored only **0.96958** public.

---

## Session 2026-06-05 Experiments Summary

### Failed experiments (do not submit)

| Script | OOF | Delta | Why failed |
|---|---|---|---|
| 29_logit_blend | 0.969071 | 0.000000 | Tied arithmetic blend; logit averaging gives no extra signal with 2 correlated base models |
| 30_meta_stacker | 0.968716 | −0.000355 | 2-model meta-learner regressed; same-feature base models lack diversity |
| 28_photometric_neighbours | 0.968931 | −0.000140 | Photometric k-NN redundant: color is already strong individual predictor → neighbourhood aggregation adds nothing |

### New candidates

| # | Submission | OOF proxy | Gate | Public risk | Notes |
|---|---|---|---|---|---|
| A | `32_spatial_5seed_blend.csv` | **0.969154** (honest) | PASSED | Proven | **Submitted: public 0.96977, current public best.** 5-seed spatial LGBM blend transferred better than its tiny OOF lift suggested |
| B | `33_loo_family.csv` (shallower) | n/a (final-only) | Final | Medium | LOO spatial + shallower LGBM (n_est=1200, num_leaves=31) × 5 seeds; class counts in public-good band; may improve public from 0.96970 |
| C | `41_5seed_lgbm_xgb_catboost.csv` | **0.969202** (honest) | PASSED | Failed public transfer | **Submitted: public 0.96958.** Best OOF did not beat public incumbent; CatBoost diversity should not be trusted without a public-risk proxy |
| D | `33_loo_family.csv` or original-data append candidate | n/a / TBD | Next | Medium | Remaining plausible lift likely needs final-feature-density transfer or new labelled rows, not another cached blend |

---

## Recommended submission order

**Do not submit `41` variants just because OOF improves.** The public score regressed despite best honest OOF.

**Primary next workstream:** original/external-data append feasibility. The forum formulae for `spectral_type` and `galaxy_population` match local train+test exactly, so the "original" dataset can be schema-aligned before append.

**Secondary public-risk probe:** `33_loo_family.csv` remains plausible because public historically rewarded LOO spatial feature density, but it has no honest OOF and should be used sparingly.

**Do NOT submit without OOF evidence:**
- `21_loo_spatial_galaxy_lean.csv`: too GALAXY-heavy
- `24_loo_spatial_stronger_nongal.csv`: lower-GALAXY direction disproved
- `28_photometric_neighbours.csv`: photometric features hurt OOF
- new `41` multiplier variants: CatBoost blend public transfer failed

---

## Class count diagnostic (reference band)
Public-good band: GALAXY 156450–156650, QSO 51250–51450, STAR 39400–39600

| Submission | GALAXY | QSO | STAR | In band? |
|---|---|---|---|---|
| 19_loo (public incumbent) | ~156538 | ~51357 | ~39540 | ✓ |
| 32_spatial_5seed | TBD (honest OOF, no forced band) | — | — | public best 0.96977 |
| 33_loo_family (shallower) | 156566 | 51275 | 39594 | ✓ |
| 41_5seed_lgbm_xgb_catboost | TBD | — | — | public regressed to 0.96958 |

---

## Key learnings from this session

1. **Logit blending ≈ arithmetic blending** when base models are highly correlated.
2. **Meta-stacking requires diverse base models** — 2 spatial models don't give enough variation.
3. **Photometric k-NN ≠ spatial k-NN**: spatial position is a WEAK individual predictor but STRONG group predictor; color is already a STRONG individual predictor, so neighborhood aggregation is redundant.
4. **More seeds helps marginally** (+0.000083 OOF for 3→5 seeds).
5. **1500-tree LGBM was already converged**; the single-fold max-tree signal did not generalize.
6. **Best honest OOF is not enough near the ceiling**: `41` beat `32` locally but lost on public by `0.00019`.
7. **Forum categorical formulae are verified locally**: `spectral_type = cut(r-g, [-inf,-1,-0.5,0,inf])`; `galaxy_population = cut(u-r, [-inf,2.2,inf])`. This makes original-data append feasible if ids/leakage/source-shift checks pass.

---

## Stop conditions for today

- Stop if two new honest-OOF candidates score below 0.969202 and do not add new information.
- Stop final-only probes if two LOO variants from new probability caches regress on public.
- Do not submit more multiplier-only variants from `19`/`25` cache.
- Do not trust CatBoost-blend OOF without a separate public-risk diagnostic.
