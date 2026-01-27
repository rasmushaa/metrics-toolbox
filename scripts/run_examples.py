#!/usr/bin/env python3
"""Execute all example notebooks using papermill.

This script executes all notebooks in the examples/ directory to ensure they run
successfully before committing changes. Notebooks are executed in place to keep outputs
updated. Papermill metadata is removed to avoid git changes.
"""

import json
import sys
from pathlib import Path

import papermill as pm


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


def main():
    """Execute all example notebooks."""
    examples_dir = Path("examples")
    notebooks = sorted(examples_dir.glob("*.ipynb"))

    if not notebooks:
        return 0

    failed = []

    for notebook in notebooks:
        try:
            # Execute notebook in place
            pm.execute_notebook(
                str(notebook),
                str(notebook),
                kernel_name="python3",
                progress_bar=False,
            )
            # Remove papermill metadata to avoid git changes
            remove_papermill_metadata(notebook)
        except Exception as e:
            print(f"ERROR: {notebook} failed - {e}")
            failed.append(notebook)

    if failed:
        print(f"\n{len(failed)} notebook(s) failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
