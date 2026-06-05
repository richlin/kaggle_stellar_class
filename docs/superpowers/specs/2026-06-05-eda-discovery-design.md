# EDA Discovery Notebook Design

## Goal

Create `notebooks/eda_discovery.ipynb`, a comprehensive exploration notebook aimed at finding the next score lift for the Kaggle stellar-classification task.

## Scope

The notebook is discovery-oriented, not a modeling pipeline. It should make the dataset understandable, then focus on where balanced-accuracy lift may still be available after the current spatial-feature breakthrough.

## Required Sections

1. Load data and validate schema from `raw_data/`.
2. Explain target balance and balanced-accuracy implications.
3. Explore numeric feature distributions by class.
4. Analyze redshift overlap and low-redshift ambiguity.
5. Analyze photometric colors and magnitudes.
6. Analyze categorical feature signal without treating categories as leakage.
7. Check train/test shift for core numeric and categorical fields.
8. Explore spatial structure: sky maps, density, and nearest-neighbor class agreement.
9. Provide residual-analysis hooks using existing experiment artifacts when available.
10. End with concrete discovery hypotheses and next experiment candidates.

## Constraints

- Use only dependencies already pinned in `requirements.txt`.
- Avoid running long model-training jobs inside the notebook.
- Keep expensive spatial diagnostics sample-aware, with clear constants at the top.
- Reuse repo modules where helpful, especially `src.data`, `src.features`, and `src.spatial`.
- The notebook must remain useful if some generated experiment artifacts are absent.

## Verification

- Add a pytest test that validates the notebook exists, contains the required section headings, and has syntactically valid Python code cells.
- Run `uv run pytest -q`.
- Run `uv run ruff check .`.
