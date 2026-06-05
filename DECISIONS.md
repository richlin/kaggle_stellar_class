# Decisions

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
