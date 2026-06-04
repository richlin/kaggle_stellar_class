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
