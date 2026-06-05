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
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for heading in REQUIRED_HEADINGS:
        assert heading in markdown


def test_eda_notebook_code_cells_compile() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    for source in code_cells:
        ast.parse(source)
