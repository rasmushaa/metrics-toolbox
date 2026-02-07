"""Test that all example notebooks execute without errors.

Add new example notebooks to the parameterized test below to ensure they are included in
the test suite.
"""

import json
from pathlib import Path

import papermill as pm
import pytest


def remove_papermill_metadata(notebook_path: Path) -> None:
    """Remove papermill metadata from notebook to avoid git changes."""
    with open(notebook_path) as f:
        notebook = json.load(f)

    # Remove papermill metadata from notebook-level metadata
    if "papermill" in notebook.get("metadata", {}):
        del notebook["metadata"]["papermill"]

    # Remove papermill and execution metadata from cell-level metadata
    for cell in notebook.get("cells", []):
        if "papermill" in cell.get("metadata", {}):
            del cell["metadata"]["papermill"]
        # Remove execution timing metadata that changes on every run
        if "execution" in cell.get("metadata", {}):
            del cell["metadata"]["execution"]

    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=1)
        f.write("\n")


@pytest.fixture
def examples_dir():
    """Return the examples directory path."""
    return Path(__file__).parent.parent / "examples"


def test_example_notebooks_exist(examples_dir):
    """Test that example notebooks exist."""
    notebooks = list(examples_dir.glob("*.ipynb"))
    assert len(notebooks) > 0, "No example notebooks found"


# Notebook names to test - add new notebooks here to include them in the test suite
@pytest.mark.parametrize(
    "notebook_name",
    [
        "binary_classification.ipynb",
        "builder.ipynb",
        "help.ipynb",
        "multiclass_classification.ipynb",
        "regression.ipynb",
    ],
)
def test_example_notebook_executes(examples_dir, notebook_name):
    """Test that each example notebook executes without errors."""
    notebook_path = examples_dir / notebook_name

    assert notebook_path.exists(), f"Notebook {notebook_name} not found"

    # Execute notebook in place
    pm.execute_notebook(
        str(notebook_path),
        str(notebook_path),
        kernel_name="python3",
        progress_bar=False,
    )

    # Remove papermill metadata to avoid git changes
    remove_papermill_metadata(notebook_path)
