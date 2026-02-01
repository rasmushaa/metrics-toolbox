#!/usr/bin/env python3
"""Pre-commit hook to update README.md with current metrics, reducers, and
requirements."""

import re
import sys
import tomllib
from pathlib import Path
from typing import List


def get_metrics() -> List[str]:
    """Extract metrics from registry enum."""
    registry_path = Path("metrics_toolbox/metrics/registry.py")
    content = registry_path.read_text()

    metrics = []
    # Find all enum entries in MetricEnum
    in_metric_enum = False

    for line in content.split("\n"):
        if "class MetricEnum(Enum):" in line:
            in_metric_enum = True
            continue
        if in_metric_enum:
            if (
                line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith('"""')
            ):
                match = re.match(r"\s*(\w+)\s*=", line)
                if match:
                    enum_name = match.group(1)
                    # Convert enum name to readable format
                    metric_name = enum_name.lower()
                    metrics.append(metric_name)
            # Stop when we reach the end of the class
            if (
                line
                and not line.startswith(" ")
                and not line.startswith("\t")
                and line.strip()
            ):
                break

    return metrics


def get_reducers() -> List[str]:
    """Extract reducers from registry enum."""
    registry_path = Path("metrics_toolbox/reducers/registry.py")
    content = registry_path.read_text()

    reducers = []
    # Find all enum entries in ReducerEnum
    in_reducer_enum = False

    for line in content.split("\n"):
        if "class ReducerEnum(Enum):" in line:
            in_reducer_enum = True
            continue
        if in_reducer_enum:
            if (
                line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith('"""')
            ):
                match = re.match(r"\s*(\w+)\s*=", line)
                if match:
                    enum_name = match.group(1)
                    reducer_name = enum_name.lower()
                    reducers.append(reducer_name)
            # Stop when we reach the end of the class
            if (
                line
                and not line.startswith(" ")
                and not line.startswith("\t")
                and line.strip()
            ):
                break

    return reducers


def get_requirements() -> List[str]:
    """Extract requirements from pyproject.toml."""
    toml_path = Path("pyproject.toml")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    dependencies = data.get("project", {}).get("dependencies", [])

    # Format requirements
    requirements = []
    for dep in dependencies:
        # Parse dependency string (e.g., "numpy>=2.0.0" or "numpy>2.0.0")
        if ">=" in dep:
            parts = dep.split(">=", 1)
            pkg = parts[0].strip()
            version = parts[1].strip()
            requirements.append(f"- {pkg} >= {version}")
        elif ">" in dep:
            parts = dep.split(">", 1)
            pkg = parts[0].strip()
            version = parts[1].strip()
            requirements.append(f"- {pkg} > {version}")
        else:
            requirements.append(f"- {dep}")

    return requirements


def update_readme() -> bool:
    """Update README.md with current metrics, reducers, and requirements."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Get current data
    metrics = get_metrics()
    reducers = get_reducers()
    requirements = get_requirements()

    # Build new sections
    metrics_section = "## Available Metrics\n\n" + "\n".join(
        f"- `{name}`" for name in metrics
    )

    reducers_section = "## Available Reducers\n\n" + "\n".join(
        f"- `{name}`" for name in reducers
    )

    requirements_section = "## Requirements\n\n- Python >= 3.11\n" + "\n".join(
        requirements
    )

    # Replace sections in README
    modified = content

    # Replace Available Metrics section
    metrics_pattern = r"## Available Metrics\n\n.*?(?=\n## |\Z)"
    modified = re.sub(
        metrics_pattern, metrics_section + "\n", modified, flags=re.DOTALL
    )

    # Replace Available Reducers section
    reducers_pattern = r"## Available Reducers\n\n.*?(?=\n## |\Z)"
    modified = re.sub(
        reducers_pattern, reducers_section + "\n", modified, flags=re.DOTALL
    )

    # Replace Requirements section
    requirements_pattern = r"## Requirements\n\n.*?(?=\n## |\Z)"
    modified = re.sub(
        requirements_pattern, requirements_section + "\n", modified, flags=re.DOTALL
    )

    # Check if content changed
    if modified != content:
        readme_path.write_text(modified)
        return True

    return False


def main():
    """Main entry point for pre-commit hook."""
    try:
        changed = update_readme()
        if changed:
            print("README.md updated with current metrics, reducers, and requirements.")
            print("Please review and stage the changes.")
            sys.exit(1)  # Exit with error to prevent commit
        else:
            print("README.md is up to date.")
            sys.exit(0)
    except Exception as e:
        import traceback

        print(f"Error updating README.md: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
