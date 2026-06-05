# Current Operating Plan: Predicting Stellar Class

This is the canonical active plan for the repo. Historical sprint plans live in
`docs/superpowers/plans/`; use them for rationale only, not as current execution
instructions.

## Current Objective

Kaggle competition: predict stellar class (`GALAXY`, `QSO`, `STAR`) using balanced
accuracy. The project is now in score-push mode, not bootstrap mode.

- **Public incumbent:** `submissions/32_spatial_5seed_blend.csv`
- **Public score:** `0.96977`
- **Target:** exceed `0.97`
- **Remaining public lift needed:** about `+0.00023`
- **Best honest OOF:** `submissions/41_5seed_lgbm_xgb_catboost.csv` at `0.969202`
- **Key caution:** `41` scored only `0.96958` public, so small OOF gains from correlated blends
  are not enough to justify more public probing.

## Source Of Truth

Read these in order at session start:

1. `PROGRESS.md` - current state, active work, blockers, and next steps.
2. `tasks/plan.md` - this current strategy and execution policy.
3. `tasks/todo.md` - active and recently completed task checklist.
4. `experiments/leaderboard.md` - public scores and local/public deltas.
5. `experiments/next_submission_report.md` - candidate ranking and submission policy.
6. `DECISIONS.md` - rationale and superseded assumptions.

## Current Strategic Read

The major lift came from leakage-safe spatial neighbourhood features. Raw `alpha` and
`delta` were weak individual predictors, but spatial k-NN class-fraction features exposed
synthetic label clustering and moved the project from the `0.966` band to the `0.969` band.

The current ceiling with existing competition-only features appears close to `0.969` to
`0.970`. Multiple attempts to squeeze the same probability caches or correlated model family
have failed to transfer enough public lift.

### What Worked

- Spatial k-NN class-fraction features on unit-sphere `(alpha, delta)`.
- Full-train/leave-one-out spatial feature density for final-only public-risk candidates.
- Extending spatial LightGBM from 3 seeds to 5 seeds:
  `32_spatial_5seed_blend.csv` became public best at `0.96977`.

### What Failed Or Is Saturated

- More class-multiplier variants from existing `19`/`25` probability caches.
- Lower-GALAXY multiplier direction (`22`) and stronger non-GALAXY variants.
- LOO-XGBoost calibration (`26`).
- Logit blending (`29`) - tied arithmetic blend.
- Two-model meta-stacking (`30`) - regressed.
- Photometric k-NN (`28`) - redundant with strong individual color features.
- More LightGBM trees (`36`) - 900-tree model was already effectively converged.
- CatBoost blend (`41`) - best honest OOF but public regressed.
- Large-k spatial features (`42`) - OOF `0.969040`, failed versus `0.969202`.

## Active Workstream

### Phase 14: Original Dataset Append

The only active high-leverage path is to test whether the original labelled dataset can add
new supervised signal. Competition discussion revealed that the two competition-only
categoricals can be recreated exactly:

```python
spectral_type = pd.cut(
    r - g,
    [-np.inf, -1, -0.5, 0, np.inf],
    labels=["M", "G/K", "A/F", "O/B"],
).astype(str)

galaxy_population = pd.cut(
    u - r,
    [-np.inf, 2.2, np.inf],
    labels=["Blue_Cloud", "Red_Sequence"],
).astype(str)
```

This was verified on all combined train/test rows and is protected by
`tests/test_data_contract.py`.

Open tasks:

- Locate and stage the original labelled dataset under a clearly named ignored path.
- Audit source, row count, schema, class labels, duplicate ids/features, leakage risk, and
  train/test/source shift.
- Train an append candidate only if the audit passes. Prefer
  `scripts/47_external_spatial_append.py` first because it uses audited original rows as
  extra labelled spatial neighbours and sweeps original-row source weights; keep
  `scripts/44_original_append_train.py` as the simpler baseline append.
- Score validation only on competition train OOF folds; original rows may be added to training
  folds but must never appear in validation folds.
- Accept only if OOF beats `0.969202` and per-class recall remains stable.
- Compare public-risk diagnostics against `32_spatial_5seed_blend.csv` before any upload.

Relevant files already implemented and guarded:

- `scripts/43_original_append_audit.py`
- `scripts/44_original_append_train.py`
- `scripts/47_external_spatial_append.py`

Do not duplicate these scripts. The current blocker is dataset acquisition/provenance
(Task 46), not script scaffolding. The audit must PASS for the same `--original` path
before append training is allowed to run.

### Phase 16: Optional New-Signal Tracks

These are implemented but blocked until inputs/dependencies exist:

- `scripts/48_tabpfn_meta_stacker.py` - optional TabPFN logit meta-stacker. Current
  environment lacks `tabpfn`, so this writes a BLOCKED ledger.
- `scripts/49_external_catalog_features.py` - external catalog feature ingestion via id or
  nearest-sky joins. Requires an allowable staged catalog CSV.

## Submission Policy

- Do not submit more `41` CatBoost-blend or multiplier-only variants without new validation
  evidence.
- Do not submit a candidate that only moves class multipliers on old caches.
- Public uploads must have:
  - entry in `experiments/leaderboard.md`;
  - candidate rationale in `experiments/next_submission_report.md`;
  - valid submission format via `src.validate` or `tests/test_submission.py`;
  - class-count and transition diagnostics against `32`.
- Stop public submissions for the day after two new candidates regress below the public
  incumbent.

## Repo Layout

- `src/` - reusable data, feature, validation, and spatial helpers.
- `scripts/` - one runnable experiment per numbered task.
- `tests/` - deterministic contract and helper tests.
- `experiments/` - JSON ledgers and probability/feature caches.
- `submissions/` - generated Kaggle CSVs.
- `notebooks/` - exploratory notebooks only; do not hide production logic here.
- `docs/superpowers/plans/` - historical sprint plans.

## Verification Gates

Before declaring work complete:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

Before considering any submission complete:

```bash
uv run python -m src.validate submissions/<candidate>.csv
```

Every score-producing experiment must write a JSON ledger under `experiments/` with:

- feature set;
- model params;
- seeds/folds;
- OOF balanced accuracy when available;
- per-class recall;
- multiplier vector;
- submission path;
- public score after submission.

## Historical Plans

Historical plans remain useful for rationale but are superseded by this file:

- `docs/superpowers/plans/2026-06-04-score-over-097-improvement.md`
- `docs/superpowers/plans/2026-06-04-transductive-spatial-task24.md`
- `docs/superpowers/plans/2026-06-05-score-over-097-revisit-plan.md`
- `docs/superpowers/plans/2026-06-05-eda-discovery-notebook.md`

If a historical plan conflicts with this file, follow this file and check `DECISIONS.md` for
the rationale.
