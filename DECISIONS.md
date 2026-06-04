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
