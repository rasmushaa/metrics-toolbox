#!/usr/bin/env python3
"""Update README.md title with current version from pyproject.toml."""

import re
import sys
import tomllib
from pathlib import Path


def main():
    """Update the README title with the current version."""
    # Read version from pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        return 1

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
        version = pyproject["project"]["version"]

    # Read README.md
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("Error: README.md not found")
        return 1

    with open(readme_path) as f:
        content = f.read()

    # Update the title with version
    # Match title with or without version
    pattern = r"^# Metrics Toolbox(?:\s+v[\d.]+)?$"
    replacement = f"# Metrics Toolbox v{version}"

    new_content, count = re.subn(
        pattern, replacement, content, count=1, flags=re.MULTILINE
    )

    if count == 0:
        print("Warning: Could not find title to update in README.md")
        return 1

    # Write back if changed
    if new_content != content:
        with open(readme_path, "w") as f:
            f.write(new_content)
        print(f"✓ Updated README.md title to version {version}")
        return 0
    else:
        print(f"✓ README.md title already at version {version}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
