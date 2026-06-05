# Decisions

## 2026-06-05 - Keep EDA Notebook Discovery-Oriented And Dependency-Light

- **Decision:** Build the EDA as a notebook using only pinned dependencies and existing repo helpers, with sample-aware spatial diagnostics instead of model-training cells.
- **Why:** The current score path depends on finding new signal, especially spatial and boundary structure. The notebook should make those signals inspectable without adding new package risk or hiding long training jobs in an exploratory artifact.
- **Applies until:** The notebook becomes a production experiment runner, at which point reusable logic should move into `src/` or `scripts/`.

## 2026-06-04 — Pure Feature Builder Plus Data Wrapper

- **Decision:** Put deterministic feature math in `src/features.py` and CSV/target handling in `src/data.py`.
- **Why:** Feature engineering can be tested with tiny fixtures, while `load_raw()` and label encoding stay close to the real Kaggle files.
- **Applies until:** Feature families become large enough to justify splitting `src/features.py` by family.

## 2026-06-04 — Fixed Competition Label Encoder

- **Decision:** Fit label encoding from the fixed class set `["GALAXY", "QSO", "STAR"]` instead of inferring from each input frame.
- **Why:** Small fixtures or future filtered folds should not change class-index mapping for metrics, prediction decoding, or experiment records.
- **Applies until:** A competition format change adds or removes target classes.

## 2026-06-04 — Direct Script Bootstrap

- **Decision:** Add minimal repo-root path bootstrapping to directly runnable scripts/modules that the plan invokes by file path.
- **Why:** `python scripts/01_baseline.py` and `python src/data.py` otherwise run with only their own directory on `sys.path`, which breaks `src.*` imports outside pytest.
- **Applies until:** The project is packaged and commands are run as installed console scripts or `python -m ...` modules.

## 2026-06-04 — Phase 1 LightGBM Baseline Parameters

- **Decision:** Use a class-weighted `LGBMClassifier` with `n_estimators=300`, `learning_rate=0.05`, `num_leaves=63`, and light row/column subsampling for the first scoreable baseline.
- **Why:** The plan prioritizes a fast, reproducible scoreable submission before CV/threshold tuning; these settings are strong enough for baseline signal without adding tuning complexity.
- **Applies until:** Phase 2/3 OOF results or Phase 4 tuning provide evidence for different parameters.

## 2026-06-04 — Stable Threshold Tuning Over Unconstrained Threshold Tuning

- **Decision:** Record the unconstrained per-class multiplier optimum but choose the best multiplier vector that respects class-recall and fold-score regression limits for `submissions/02_cv_tuned.csv`.
- **Why:** The unconstrained vector improved OOF balanced accuracy but dropped GALAXY recall by about `0.0076`; the chosen vector `[0.75, 0.75, 0.9]` still improves OOF while keeping the worst class recall delta around `-0.00292` and worst fold delta around `-0.00006`.
- **Applies until:** A later repeated-seed or public leaderboard comparison shows that a different stability threshold is justified.

## 2026-06-04 — Repeated-Seed Averaging Beats Extra Parameter Complexity

- **Decision:** Keep the Phase 3-like LightGBM parameters for `submissions/03_final.csv`, but average OOF/test probabilities across seeds `42`, `43`, and `44`, then re-tune stable class multipliers to `[0.9, 0.8, 1.15]`.
- **Why:** Manual parameter screening found `phase3_like` best (`0.9655647551693756` single-seed OOF). The repeated-seed average improved to `0.9659249816190973`, beating the Phase 3 reference without increasing model complexity.
- **Applies until:** A later ensemble or public leaderboard result provides stronger evidence for a different model family or parameter set.

## 2026-06-04 — Keep Baseline Feature Set After Ablation

- **Decision:** Do not remove feature families for the final model despite some 3-fold ablations showing tiny positive deltas when dropped.
- **Why:** The positive ablation deltas for dropping raw magnitudes, magnitude summaries, and redshift interactions were small 3-fold screening effects, while coordinate features were clearly required and color/categorical interaction drops were neutral-to-negative. Keeping the full baseline feature set is the conservative choice for repeated-seed final validation.
- **Applies until:** Repeated 5-fold ablation evidence shows a feature-family removal improves OOF without destabilizing per-class recall.

## 2026-06-04 — Phase 5 Blend Keeps Only Useful Diversity

- **Decision:** Use the Phase 5 blend `0.7 * lgbm_seed_average_final + 0.3 * xgboost`, with CatBoost and LightGBM DART retained as audited zero-weight candidates.
- **Why:** XGBoost was weaker standalone (`0.965349` tuned OOF) but improved the blend to `0.9660551711148327`. CatBoost (`0.962913`) and DART (`0.963829`) were lower standalone and did not improve the best OOF blend, so the weight search rejected them.
- **Applies until:** A public leaderboard result or targeted boundary-feature experiment shows a different blend improves balanced accuracy without breaching the fold/class recall guardrails.

## 2026-06-04 — Continuous Threshold Search Must Preserve Stable Grid Baseline

- **Decision:** Seed Phase 5 continuous multiplier optimization from the exhaustive stable threshold grid before running Nelder-Mead.
- **Why:** Coordinate-ascent seeding regressed the reference-only model toward argmax. The stable grid reproduces the Phase 4 guardrail semantics, and continuous optimization then adds a small local gain without weakening fold or class recall constraints.
- **Applies until:** A faster optimizer can prove equal or better guarded OOF scores against the same saved probability arrays.

## 2026-06-04 — Public Best Overrides Local Micro-Gains

- **Decision:** Keep `submissions/03_final.csv` as the incumbent public-best submission after `04_ensemble.csv` scored `0.96676` versus `03_final.csv` at `0.96691`.
- **Why:** `04_ensemble` improved local OOF by only `+0.000130` while moving mostly prior `STAR` predictions to `GALAXY`/`QSO`. The public result shows that narrow threshold trade did not transfer.
- **Applies until:** A new public submission beats `0.96691`.

## 2026-06-04 — STAR-Safe Blend Is Preferred Over Aggressive Ensemble Thresholds

- **Decision:** Use `submissions/06_star_safe_blend.csv` as the next public candidate instead of `04_ensemble.csv` or `05_boundary_v1.csv`.
- **Why:** The STAR-safe blend has stronger local OOF (`0.9662339484478014`) than both `03_final` and `04_ensemble`, while nearly preserving STAR recall versus `03_final` (`-0.00010879551278952793`) after public evidence showed STAR-boundary movement was risky. `boundary_v1` failed the local gate at `0.9656299274771866`.
- **Applies until:** Its public score is known or a stronger candidate improves both OOF and public-risk profile.

## 2026-06-04 — Reject Stacking, Pseudo-Labeling, And Class-Weight Variants Without New Evidence

- **Decision:** Do not submit `07_probability_stacker.csv`, `08_pseudolabel.csv`, or standalone class-weight variant outputs as primary candidates.
- **Why:** The probability stacker failed the local gate (`0.965915` tuned OOF). Unweighted and square-root-balanced LightGBM variants lifted GALAXY only by sacrificing minority recall and scored `0.959421` / `0.963588`. The pseudo-label candidate has no honest OOF score and lowers STAR submission count to `39,008`, matching the public-regression pattern from `04_ensemble`.
- **Applies until:** A new public score or a redesigned validation proxy shows these paths transfer better than the current evidence suggests.

## 2026-06-04 — Extended Seed Average Is Secondary To STAR-Safe Blend

- **Decision:** Keep `09_extended_seed_average.csv` as a secondary, lower-risk candidate, not the primary next submission.
- **Why:** Extending the LightGBM average from 3 to 5 seeds improved over `03_final` locally (`0.966006` vs `0.965925`) but remains below `06_star_safe_blend` (`0.966234`) and reduces STAR count more than the incumbent.
- **Applies until:** `06_star_safe_blend.csv` public score is known or additional seed averaging changes the risk/reward profile.

## 2026-06-04 — Defer Longer XGBoost Tuning Until Public Feedback

- **Decision:** Do not run the full 120-trial XGBoost tuning plan before submitting `06_star_safe_blend.csv`.
- **Why:** An 8-trial Optuna run with early stopping improved tuned XGBoost argmax OOF to `0.965309`, but the resulting blend scored only `0.966062`, below `06_star_safe_blend` at `0.966234`, and moved more prior STAR predictions away from STAR. More local tuning may overfit the same weakly transferring boundary.
- **Applies until:** `06_star_safe_blend.csv` public score is known or a revised validation target better predicts public movement.

## 2026-06-04 — Target Encoding Helps Only As Small Blend Diversity

- **Decision:** Submit `submissions/11_target_encoding_blend.csv` before `06_star_safe_blend.csv`, but do not submit standalone `10_target_encoding.csv`.
- **Why:** The standalone target-encoding LightGBM failed the local gate (`0.965530`), but adding it at 10% weight to the reference/XGBoost blend improved OOF to `0.9662595764879448`, slightly above `06_star_safe_blend`, while preserving STAR recall versus `03_final` (`-0.00007253034185961127`).
- **Applies until:** Its public score is known or a stronger candidate improves both OOF and public-risk profile.

## 2026-06-04 — Multi-Blend Supersedes Target-Encoding Blend Locally

- **Decision:** Submit `submissions/12_multi_blend.csv` before `11_target_encoding_blend.csv`.
- **Why:** A cached-probability blend of `03_final`, XGBoost, extended seed averaging, and `boundary_v1` improved local OOF to `0.9662824834818386`, slightly above `11_target_encoding_blend` at `0.9662595764879448`. It changes fewer test rows than `11` versus `03_final` (793 vs 886), though its STAR count is lower, so this remains a public-risk candidate rather than a verified new incumbent.
- **Applies until:** Its public score is known or a stronger candidate improves both OOF and public-risk profile.

## 2026-06-04 — Multi-Blend Is The Public Incumbent

- **Decision:** Treat `submissions/12_multi_blend.csv` as the public-best fallback and deprioritize `13_class_weight_lgbm.csv`.
- **Why:** Public results from Kaggle show `12_multi_blend.csv` at `0.96711`, ahead of `11_target_encoding_blend.csv` (`0.96700`), `13_class_weight_lgbm.csv` (`0.96692`), `03_final.csv` (`0.96691`), `04_ensemble.csv` (`0.96676`), `10_target_encoding.csv` (`0.96673`), and `09_extended_seed_average.csv` (`0.96658`). The remaining target gap is about `+0.00289`, too large for more cached blend micro-tuning to be a credible primary path.
- **Applies until:** A new public submission beats `0.96711`.

## 2026-06-04 — Local OOF > 0.97 Is Above The Feature Ceiling; Stop Model-Search Hill-Climbing

- **Decision:** Do not pursue the `> 0.97` local OOF target via further model/hyperparameter/blend search. Any future push must add **new information** (morphology / extra bands / external catalogs), not more model tuning. Diagnostic recorded in `scripts/14_ceiling_diagnosis.py` and `experiments/14_ceiling_diagnosis.json`.
- **Why:** Convergent evidence, not a single estimate. (1) 13 diverse strong approaches span only `0.00075` local OOF (best `0.966282`); the target is `+0.0037`, ~5× that spread. (2) The residual error is **structural**: 84% of GALAXY→STAR errors sit at `redshift < 0.15` (median `0.073`), where redshift — the dominant feature — is ≈0 for both stars and low-z galaxies and loses discriminating power; the physical tie-breaker SDSS uses (morphology: point-source vs extended) is **absent** from these 10 features. (3) ~28% of errors are `>0.90`-confident-wrong (label-noise / feature-aliasing), which caps *measured* accuracy regardless of model. (4) Negative controls: a k-NN Bayes estimator under-performs the models themselves (~0.90–0.92 even redshift-weighted), so it cannot bound the ceiling; and only `246 / 577,348` rounded-feature cells carry conflicting labels, so there is no hard pigeonhole wall either — the ceiling is informational, set by feature content.
- **Applies until:** New features encoding morphology/structure (or additional photometric bands / cross-matched catalogs) are added to the dataset, at which point the Bayes floor for the low-redshift GALAXY/STAR boundary should be re-estimated.

## 2026-06-04 — SUPERSEDED: The "0.966 Ceiling" Was Wrong — Spatial Structure Is The Gap

- **Decision:** Reverse the decision above. The `0.966` ceiling claim was **incorrect**; pursue spatial neighbourhood features as the primary lever. `submissions/16_spatial_blend.csv` (OOF `0.969071`) is the new top local candidate.
- **Why it was wrong:** The leaderboard shows `0.97+` is achieved by many teams (top `0.97127`) — direct disconfirmation. The error in reasoning: model convergence at `0.966` reflected a **shared representational blind spot**, not a Bayes floor. Gradient-boosted trees on raw `alpha`/`delta` can only split axis-aligned and cannot encode "fraction of my spatial neighbours that are class c". The k-NN "ceiling estimate" scoring *below* the models should have been read as "the estimator is weak", which it was.
- **What actually works:** Class is clustered in sky position (position-only LightGBM `0.678` bal-acc; 10-NN same-class rate `0.68` vs `0.49` chance — synthetic spatial structure). Out-of-fold k-NN class-fraction features on unit-sphere `(alpha,delta)` lift OOF from `0.966282` to `0.968894` (LightGBM, `scripts/15_spatial_features.py`) and `0.969071` blended with a spatial-aware XGBoost (`scripts/16_spatial_xgb.py`). STAR recall `0.952→0.975`.
- **Leakage controls (mandatory):** Spatial features use KFold-OOF on train (a row's features never use its own fold; unit-tested in `tests/test_spatial_features.py`) and full-train→test for the test set. A read-only pre-check confirmed test points are spatially intermixed with train (matched nearest-train-neighbour distance distributions), so the signal transfers and is not encoded leakage.
- **Rejected sub-options:** adding redshift to the neighbour metric *hurt* (`0.9665` vs `0.9686` single-split) — it dilutes the pure positional signal; finer k (`5,10`) helped a single split but not full CV.
- **Open / next:** GALAXY recall (`0.958`) is now the binding constraint; remaining gap to the `0.9708` leader cluster (~`0.002` OOF) likely needs even finer spatial encoding (position-cell target encoding) or a spatial signal we have not yet captured. **The real gate is the Kaggle public score on `16_spatial_blend.csv`** — adopt as incumbent over `12_multi_blend` only if public beats `0.96711`.
- **Applies until:** Public score on `16_spatial_blend.csv` is known.

## 2026-06-04 — Spatial Blend Is The Public Incumbent

- **Decision:** Treat `submissions/16_spatial_blend.csv` as the new public-best submission.
- **Why:** Kaggle public score is `0.96927`, improving `+0.00216` over the prior public incumbent `12_multi_blend.csv` (`0.96711`). This confirms the spatial-neighbour feature transfer and narrows the remaining gap to the target `>0.97` to about `+0.00073`.
- **Applies until:** A later public submission beats `0.96927`.

## 2026-06-04 — Reject Broad Graph/Cluster Residuals Unless Public Evidence Contradicts OOF

- **Decision:** Do not submit `submissions/17_transductive_spatial.csv` or `submissions/18_galaxy_residual.csv`.
- **Why:** `17_transductive_spatial.csv` added graph probabilities, multi-resolution cluster rates, and probability meta-features, but its residual model tuned OOF was only `0.968377`; blend search assigned residual weight `0.0`, leaving the exact `16_spatial_blend` predictions. `18_galaxy_residual.csv` trained binary GALAXY residual models inside incumbent STAR/QSO regions, but threshold search selected no flips and OOF remained `0.969071`. The residual top precision was below the balanced-accuracy break-even point for flipping STAR/QSO predictions to GALAXY.
- **Applies until:** A new residual feature source proves it can flip GALAXY misses with enough precision to beat `16` on OOF or public leaderboard.

## 2026-06-04 — Use LOO Spatial Final Variants As Public-Risk Probes

- **Decision:** Keep `16_spatial_blend.csv` as the incumbent, but if submission slots are available, probe `20_loo_spatial_neutral.csv` before the higher-risk `21_loo_spatial_galaxy_lean.csv`.
- **Why:** The top-10 leaderboard cluster above `0.9707` suggests the remaining edge may be final train/test spatial-feature density, not another honest OOF residual. Existing spatial models train on KFold-OOF spatial features while test features use all train labels. `scripts/19_loo_spatial_final.py` trains the LightGBM component on leave-one-out spatial features, then blends with the existing spatial XGBoost component. This has no honest OOF score, so it is public-risk only. `20` is near GALAXY-neutral versus `16` (`+38` GALAXY, 441 changed rows); `21` is GALAXY-leaning (`+835` GALAXY, 890 changed rows) and should only follow if a more conservative probe helps.
- **Applies until:** Public leaderboard feedback shows whether LOO final-feature density transfers.

## 2026-06-04 — LOO Spatial Transfer Is Real; Stop GALAXY-Lean Probes For Now

- **Decision:** Treat `submissions/19_loo_spatial_final.csv` as the public incumbent and prioritize `23_loo_spatial_star_tilt.csv`, then `22_loo_spatial_mild_nongal.csv`, before any GALAXY-leaning variant.
- **Why:** Public scores show `19_loo_spatial_final.csv` at `0.96970` and `20_loo_spatial_neutral.csv` at `0.96968`, both above `16_spatial_blend.csv` at `0.96927`. This confirms the final-feature-density hypothesis. It also disproves the immediate GALAXY-lean submission policy: the lower-GALAXY `19` beat the GALAXY-neutral `20`, so the next probes should stay near `19` and cautiously test more STAR/non-GALAXY movement. The remaining gap to `0.97` is only `+0.00030`.
- **Applies until:** Public scores for `22`/`23` show whether the non-GALAXY direction continues or reverses.

## 2026-06-04 — STAR Tilt Tied The Incumbent; Test Mild Non-GALAXY Next

- **Decision:** Keep `19_loo_spatial_final.csv` as the public incumbent and submit `22_loo_spatial_mild_nongal.csv` before `24_loo_spatial_stronger_nongal.csv`.
- **Why:** `23_loo_spatial_star_tilt.csv` scored `0.96970`, tying `19` but not improving. That rules out continuing the small STAR-tilt direction as the next best use of a submission slot. The remaining plausible axis is the lower-GALAXY/non-GALAXY movement represented by `22`; `24` is a stronger version and should wait for `22` feedback.
- **Applies until:** Public score for `22_loo_spatial_mild_nongal.csv` is known.

## 2026-06-04 — Lower-GALAXY Multiplier Direction Failed; Test Calibrated LOO XGBoost

- **Decision:** Do not submit `24_loo_spatial_stronger_nongal.csv`. Submit `26_loo_spatial_xgb_calibrated.csv` next if a slot is available.
- **Why:** `22_loo_spatial_mild_nongal.csv` scored `0.96944`, below `19`/`23` at `0.96970`, so the lower-GALAXY multiplier direction is now disproven. The next independent axis is training the XGBoost blend component on LOO spatial features as well. Raw `25` was too GALAXY-heavy, so `26` calibrates its class counts back near the public-best `19` distribution while changing a different set of rows.
- **Applies until:** Public score for `26_loo_spatial_xgb_calibrated.csv` is known.

## 2026-06-05 — Cache-Level Probing Is Saturated; Revisit With New Signal

- **Decision:** Keep `19_loo_spatial_final.csv` / `23_loo_spatial_star_tilt.csv` as the public incumbent at `0.96970` and stop submitting variants derived only from the current `19` or `25` cached probability files.
- **Why:** The latest public result, `26_loo_spatial_xgb_calibrated.csv = 0.96956`, regressed despite matching the `19` class mix more closely. The submission history has bracketed the easy axes: GALAXY-neutral (`20`) is slightly worse, lower-GALAXY (`22`) is worse, STAR tilt (`23`) ties, and LOO-XGBoost (`26`) is worse. The remaining `+0.00030` likely requires a new spatial signal, a better final-feature validation proxy, or a materially different model component.
- **Applies until:** A new plan produces a candidate not reducible to class-multiplier movement on existing cached probabilities.

## 2026-06-05 — Photometric Neighbourhood Features Are The Next Signal Source

- **Decision:** Implement k-NN class-fraction features in three photometric spaces (colour 4D, magnitude 5D, sphere+colour 7D) as the primary OOF-lift experiment for 2026-06-05.
- **Why:** Feature MI analysis shows photometric k-NN features have mean MI=0.301 with class labels vs spatial features' mean MI=0.174 — 73% higher. The top features are magnitude-space k-NN fractions (not colour), because magnitude clusters objects by absolute brightness. Cross-correlation with existing spatial features is only 0.60, confirming substantially independent information. Individual colour features are already in the model; the neighbourhood-level aggregation is the new signal.
- **Applies until:** OOF result from `scripts/28_photometric_neighbours.py` is known.

## 2026-06-05 — Logit Blend And 2-Model Meta-Stacker Do Not Improve On Spatial Blend

- **Decision:** Do not submit logit-blend (`scripts/29_logit_blend.py`) or 2-model meta-stacker (`scripts/30_meta_stacker.py`) variants.
- **Why:** Logit blend (geometric mean) tied arithmetic blend exactly at OOF=0.969071. The 2-model meta-stacker scored 0.968716 (−0.000355 vs incumbent). Both failures have the same root cause: with only 2 highly correlated base models (both trained on spatial features), neither the blending method nor the meta-learner can add new information. The meta-stacker regression likely occurs because the LightGBM meta-learner with averaged (multi-seed) OOF features has less discriminative power than the direct optimised blend.
- **Applies until:** A third base model with materially different features (photometric LGBM from script 28) is available for meta-stacking (see script 34).

## 2026-06-05 — Photometric Neighbourhood Features Do Not Improve OOF

- **Decision:** Do not submit `submissions/28_photometric_neighbours.csv` (gate failed). Also deprioritize scripts 31, 34, 35, 37 (all depend on photometric OOF probs that hurt OOF).
- **Why:** Full 5-fold × 3-seed result: tuned OOF `0.968931` (−0.000140 vs incumbent `0.969071`). Root cause — photometric k-NN is redundant: individual color features (u_g, g_r, r_i, i_z, u, g, r, i, z) are already STRONG class predictors in the model. Adding neighborhood aggregates of the same color signal adds no new discriminative power. This is the opposite of spatial features: (alpha, delta) as individual features are WEAK class predictors, so their k-NN class-fraction aggregation IS genuinely new information.
- **Key learning for future sessions:** k-NN neighborhood features add value only when the underlying coordinates are weak individual predictors. Strong individual features do not benefit from neighborhood aggregation.
- **Applies until:** A new photometric feature space is identified that is (a) weakly predictive as individual features but (b) strongly predictive as neighborhood aggregates.

## 2026-06-05 — 1500-Tree Spatial LGBM As The Remaining OOF-Lift Path

- **Decision:** Focus remaining session compute on `scripts/36_spatial_more_trees.py` (n_estimators=1500) and the LOO family (script 33) for public score.
- **Why:** Single-fold comparison showed BOTH spatial-only and spatial+photometric models hit max iterations at n_estimators=900 (`best_iter=899/900`). This means the model was still improving and was capacity-constrained. Increasing to 1500 trees should provide additional lift. Expected improvement from 900→1500 trees: +0.0005 to +0.0015. Combined with 5-seed extension (+0.000083 from script 32), total expected new OOF range: 0.9697–0.9707.
- **Applies until:** Script 36 OOF result is known.

## 2026-06-05 — 1500-Tree LGBM Does Not Improve OOF; Model Was Already Converged

- **Decision:** Do not pursue further tree-count increases (script 37, 38, 39). The 900-tree spatial LGBM is effectively converged.
- **Evidence:** 1500-tree spatial LGBM standalone OOF = 0.968904 (vs 900-tree 0.968894 = +0.000010 difference). Best 3-model blend OOF = 0.969044, BELOW the 5-seed spatial blend 0.969154. Gate FAILED.
- **Why the single-fold evidence was misleading:** The spatial-only single fold (n_est=900) showed best_iter=900 (hit max). But across the full 5-fold × 3-seed ensemble, most folds trigger early stopping well before 900 trees. The one fold that hit max had high variance and did not represent the average behaviour.
- **Implication:** The 900-tree OOF ceiling for spatial LGBM+XGBoost blend is ~0.969154 (5-seed). Further tree-count or learning-rate experiments are not expected to improve this.
- **Remaining open path:** CatBoost with spatial features (script 40, running) — genuinely different model family (symmetric oblivious trees, ordered boosting). If standalone OOF > 0.967 and it provides sufficient diversity, 3-model blend might reach ~0.970+.
- **Applies until:** CatBoost OOF result is known.

## 2026-06-05 — Spatial 5-Seed Is Public Incumbent; CatBoost Blend Did Not Transfer

- **Decision:** Treat `submissions/32_spatial_5seed_blend.csv` as the public incumbent at `0.96977`, and do not pursue CatBoost-blend multiplier variants from `41` without new validation evidence.
- **Why:** `32_spatial_5seed_blend.csv` improved public score from `0.96970` to `0.96977`. `41_5seed_lgbm_xgb_catboost.csv` had the best honest OOF (`0.969202`) but scored only `0.96958` public, below both `32` and the prior `19`/`23` incumbent. Near this ceiling, small OOF gains from correlated blend diversity are not reliable enough for public probing.
- **Applies until:** A new validation proxy predicts public movement better than OOF, or a new feature/source adds materially different signal.

## 2026-06-05 — Original Dataset Append Is Feasible If Categoricals Are Recreated

- **Decision:** Open a new workstream to test appending the original labelled dataset only after recreating `spectral_type` and `galaxy_population`, then passing duplicate, leakage, and source-shift checks.
- **Why:** The competition discussion formulae match local train+test exactly: `spectral_type` is a thresholded `r-g` feature and `galaxy_population` is a thresholded `u-r` feature. That removes the schema mismatch that previously made the original dataset hard to append. This is genuinely new labelled data, unlike more threshold/cache probing.
- **Applies until:** The original dataset cannot be obtained, fails schema/leakage checks, or an honest competition-train OOF experiment shows appended data does not improve the public-incumbent path.

## 2026-06-05 — Tasks Plan Is The Canonical Current Operating Plan

- **Decision:** Replace `tasks/plan.md` with a current operating plan and treat `docs/superpowers/plans/` as historical sprint-plan archive.
- **Why:** The old `tasks/plan.md` still described the bootstrap state and early LightGBM phases, while the real project state had moved to `PROGRESS.md`, `DECISIONS.md`, `experiments/leaderboard.md`, and scattered sprint plans. Future agents need one active plan to avoid executing stale instructions.
- **Applies until:** The project changes from Kaggle score-push mode to a different objective, at which point `tasks/plan.md` should be rewritten again rather than extended indefinitely.

## 2026-06-05 — Galactic Coordinate Features Add Minimal Independent Signal

- **Decision:** Do not submit `submissions/46_4model_galactic_blend.csv` as a next Kaggle upload — the OOF improvement is +0.000009, too small to override the observed pattern that additional model complexity hurts public transfer.
- **Evidence:** Galactic+spatial LGBM standalone OOF 0.968967 (+0.000073 vs spatial-only 0.968894). Best 4-model blend: 0.969211 (+0.000009 vs 3-model incumbent 0.969202). Per-class recalls confirm marginal movement. Key finding: 23.8% of GALAXY→STAR errors are at |b|>60° (where stars are rare) — these objects SHOULD have been captured by the galactic latitude feature. That they weren't captured effectively confirms the existing spatial k-NN fractions already encode most of the galactic structure information (high-|b| sky regions have high GALAXY fraction in the k-NN neighborhood, so the model already implicitly uses this signal).
- **Why so little improvement despite strong galactic latitude signal:** The spatial k-NN class fractions at k=50..250 largely capture sky-region class distributions, which correlates with galactic latitude. Adding b directly gives the model a shortcut, but the model had already learned this pattern through the k-NN features.
- **Applies until:** A substantially different spatial feature strategy (e.g., very wide k or entirely different coordinate decomposition) is designed specifically to capture multi-scale galactic structure not encoded by current k∈{5..250} fractions.

## 2026-06-05 — OOF Ceiling Confirmed At ~0.969-0.970 For Tree Models On Current Features

- **Decision:** Stop feature engineering experiments unless a qualitatively new data source (original SDSS append) or approach (neural networks) becomes available.
- **Evidence from exhaustive experiments (all within this session):**
  | Approach | OOF delta | Result |
  |---|---|---|
  | +2 seeds (3→5) | +0.000083 | Modest, transfers well |
  | Photometric k-NN (3 spaces) | −0.000140 | Color = strong individual predictor |
  | 1500-tree LGBM | +0.000010 | Model already converged at 900 trees |
  | 2-model meta-stacker | −0.000355 | Needs model diversity |
  | CatBoost diversity | +0.000048 OOF, −0.00019 public | Doesn't transfer |
  | Large-k spatial (k=1000,5000) | −0.000162 | Approaches global prior |
  | Galactic coordinates | +0.000073 standalone | Partially redundant with spatial k-NN |
  | 4-model blend | +0.000009 | All diversity sources exhausted |
- **Best honest OOF achieved:** 0.969211 (4-model LGBM5 + XGB + CatBoost + Galactic).
- **Gap to 0.971:** 0.001789 — requires fundamentally new information.
- **Remaining viable path:** Original data append (Tasks 46-48): train on competition train + SDSS spectroscopic catalog (same schema after categorical formula derivation), with OOF evaluated only on competition rows.
- **Applies until:** Original SDSS dataset is located and passes the audit in `scripts/43_original_append_audit.py`.

## 2026-06-05 — Original-Append Scaffolding Must Fail Closed

- **Decision:** Treat the original-data audit/train scripts as ready scaffolding, but make them fail closed before any dataset run: categorical formula mismatches fail audit, exact or 6-decimal feature duplicates fail audit, and append training refuses to run unless `--original` matches the PASS audit's `original_path`.
- **Why:** The original-data append is the only remaining high-leverage path, but it has the highest leakage/provenance risk. A stale PASS audit or a duplicate feature row could create an invalid leaderboard gain, so the guardrails need to reject ambiguous inputs before modeling starts.
- **Applies until:** A staged original dataset passes the hardened audit and produces an append candidate; then the same checks should remain as regression coverage.
