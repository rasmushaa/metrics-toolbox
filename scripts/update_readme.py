#!/usr/bin/env python3
"""Update README.md title with current version from pyproject.toml.

and the requirements.txt files.
"""

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
        dependencies = pyproject["project"]["dependencies"]

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

    # Update requirements section
    requirements_section = (
        "## Requirements\n\n" + "\n".join(f"- {dep}" for dep in dependencies) + "\n\n"
    )

    # Check if Requirements section exists
    requirements_pattern = r"## Requirements\n\n(?:- [^\n]+\n)+\n"

    if re.search(requirements_pattern, new_content):
        # Replace existing Requirements section
        new_content = re.sub(
            requirements_pattern, requirements_section, new_content, count=1
        )
    else:
        # Insert Requirements section after the badges (before ## Features)
        features_pattern = r"(\n## Features)"
        if re.search(features_pattern, new_content):
            new_content = re.sub(
                features_pattern, f"\n{requirements_section}\\1", new_content, count=1
            )
        else:
            print("Warning: Could not find location to insert Requirements section")
            return 1

    # Write back if changed
    if new_content != content:
        with open(readme_path, "w") as f:
            f.write(new_content)
        print(f"✓ Updated README.md title to version {version}")
        print(f"✓ Updated README.md requirements section")
        return 0
    else:
        print(f"✓ README.md title already at version {version}")
        print(f"✓ README.md requirements already up to date")
        return 0


if __name__ == "__main__":
    sys.exit(main())
