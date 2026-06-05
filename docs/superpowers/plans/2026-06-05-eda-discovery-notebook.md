# EDA Discovery Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive discovery EDA notebook for finding the next balanced-accuracy lift.

**Architecture:** The notebook is a single durable artifact under `notebooks/`, backed by a lightweight pytest structure test. It uses pinned dependencies and existing repo helpers, with sampling constants to keep expensive exploratory cells runnable on a laptop.

**Tech Stack:** Jupyter notebook JSON via `nbformat`, pandas, numpy, matplotlib, scikit-learn, and existing `src.*` modules.

---

## File Structure

- Create `notebooks/eda_discovery.ipynb`: the EDA artifact, with markdown narrative and runnable code cells.
- Create `tests/test_eda_notebook.py`: structure and syntax checks for the notebook.
- Create `docs/superpowers/specs/2026-06-05-eda-discovery-design.md`: approved design.
- Create `docs/superpowers/plans/2026-06-05-eda-discovery-notebook.md`: this implementation plan.
- Modify `PROGRESS.md`: record the completed notebook and verification results at session end.
- Modify `DECISIONS.md`: record the dependency and execution-scope choices.

## Task 1: Notebook Contract Test

**Files:**
- Create: `tests/test_eda_notebook.py`
- Target artifact: `notebooks/eda_discovery.ipynb`

- [x] **Step 1: Write the failing test**

```python
from __future__ import annotations

import ast
from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("notebooks/eda_discovery.ipynb")
REQUIRED_HEADINGS = [
    "# Stellar Class Discovery EDA",
    "## 1. Load Data And Validate Schema",
    "## 2. Target Balance And Metric Implications",
    "## 3. Numeric Feature Distributions By Class",
    "## 4. Redshift Overlap And Low-Redshift Ambiguity",
    "## 5. Photometric Color And Magnitude Relationships",
    "## 6. Categorical Signal",
    "## 7. Train/Test Shift Checks",
    "## 8. Spatial Structure Discovery",
    "## 9. Residual Analysis Hooks",
    "## 10. Discovery Hypotheses And Next Experiments",
]


def test_eda_notebook_exists_with_required_sections() -> None:
    assert NOTEBOOK_PATH.exists()
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    for heading in REQUIRED_HEADINGS:
        assert heading in markdown


def test_eda_notebook_code_cells_compile() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    for source in code_cells:
        ast.parse(source)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eda_notebook.py -q`

Expected: failure because `notebooks/eda_discovery.ipynb` does not exist yet.

## Task 2: Discovery Notebook

**Files:**
- Create: `notebooks/eda_discovery.ipynb`

- [x] **Step 1: Generate the notebook**

Use `nbformat` to create the notebook with the required markdown sections and runnable code cells. Include these constants near the top:

```python
RANDOM_STATE = 42
PLOT_SAMPLE_ROWS = 80_000
SPATIAL_SAMPLE_ROWS = 120_000
NN_SAMPLE_ROWS = 30_000
```

- [x] **Step 2: Include discovery cells**

The notebook must include code that:

```python
train, test, sample_submission = load_raw()
X_train, y_train, categorical_columns, encoder = build_features(train)
X_test, _, _, _ = build_features(test, label_encoder=encoder)
```

and cells for schema checks, class balance, distribution plots, redshift ambiguity, color features, categorical crosstabs, train/test shift tables, spatial scatter plots, nearest-neighbor class agreement, and optional experiment artifact loading.

- [x] **Step 3: Run structure test**

Run: `uv run pytest tests/test_eda_notebook.py -q`

Expected: pass.

## Task 3: Repo Tracking And Verification

**Files:**
- Modify: `PROGRESS.md`
- Modify: `DECISIONS.md`

- [x] **Step 1: Record progress**

Append a completed entry to `PROGRESS.md`:

```markdown
- 2026-06-05: Added `notebooks/eda_discovery.ipynb`, a discovery-oriented EDA notebook focused on class balance, redshift ambiguity, train/test shift, spatial clustering, residual-analysis hooks, and next experiment hypotheses.
```

- [x] **Step 2: Record decision**

Append a decision to `DECISIONS.md`:

```markdown
## 2026-06-05 - Keep EDA Notebook Discovery-Oriented And Dependency-Light

- **Decision:** Build the EDA as a notebook using only pinned dependencies and existing repo helpers, with sample-aware spatial diagnostics instead of model-training cells.
- **Why:** The current score path depends on finding new signal, especially spatial and boundary structure. The notebook should make those signals inspectable without adding new package risk or hiding long training jobs in an exploratory artifact.
- **Applies until:** The notebook becomes a production experiment runner, at which point reusable logic should move into `src/` or `scripts/`.
```

- [x] **Step 3: Run full verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
```

Expected: all tests pass and ruff reports no issues.
