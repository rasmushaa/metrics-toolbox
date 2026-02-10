#!/usr/bin/env bash

# Script to test locally with lowest-direct dependencies
# This mirrors the CI lowest-deps test configuration

set -e  # Exit on any error

echo "=========================================="
echo "Testing with lowest-direct dependencies"
echo "=========================================="

# Clean any existing environment
echo "Cleaning existing environment..."
rm -rf .venv-lowest

# Create fresh environment with lowest dependencies
echo "Creating virtual environment with lowest dependencies..."
uv venv .venv-lowest
source .venv-lowest/bin/activate

# Install with lowest-direct resolution (same as CI)
echo "Installing dependencies with lowest-direct resolution..."
uv sync --resolution=lowest-direct

# Show what was installed
echo ""
echo "Installed packages:"
uv pip list

# Run tests
echo ""
echo "Running tests..."
uv run pytest

echo ""
echo "✓ Tests passed with lowest-direct dependencies!"
echo "To continue using this environment: source .venv-lowest/bin/activate"
