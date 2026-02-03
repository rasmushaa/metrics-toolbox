#!/usr/bin/env python3
"""Update Python version badge in README.md based on pyproject.toml classifiers."""

import re
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from pathlib import Path


def get_python_versions_from_toml() -> list[str]:
    """Extract Python versions from pyproject.toml classifiers."""
    toml_path = Path("pyproject.toml")
    if not toml_path.exists():
        print("❌ pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    versions = []
    for classifier in data["project"].get("classifiers", []):
        if classifier.startswith("Programming Language :: Python :: 3."):
            version = classifier.split("::")[-1].strip()
            versions.append(version)

    if not versions:
        print("❌ No Python version classifiers found", file=sys.stderr)
        sys.exit(1)

    return sorted(versions)


def format_version_range(versions: list[str]) -> str:
    """Format version list into a readable range string."""
    if not versions:
        return ""
    if len(versions) == 1:
        return versions[0]
    return f"{versions[0]}%E2%80%93{versions[-1]}"  # %E2%80%93 is URL-encoded en-dash


def update_readme(versions: list[str]) -> bool:
    """Update the Python version badge in README.md."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("❌ README.md not found", file=sys.stderr)
        sys.exit(1)

    content = readme_path.read_text()
    version_range = format_version_range(versions)

    # Pattern to match the Python version badge
    pattern = r"\[!\[Python Version\]\(https://img\.shields\.io/badge/python-[^\)]+\)\]"
    new_badge = f"[![Python Version](https://img.shields.io/badge/python-{version_range}-blue.svg)]"

    new_content = re.sub(pattern, new_badge, content)

    if new_content == content:
        print("✓ Python version badge already up to date")
        return False

    readme_path.write_text(new_content)
    print(f"✓ Updated Python version badge to: {', '.join(versions)}")
    return True


def main():
    """Main entry point."""
    versions = get_python_versions_from_toml()
    changed = update_readme(versions)

    if changed:
        sys.exit(1)  # Signal pre-commit that file was modified
    sys.exit(0)


if __name__ == "__main__":
    main()
