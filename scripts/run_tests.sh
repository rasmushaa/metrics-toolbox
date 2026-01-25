#!/usr/bin/env bash
# Custom pre-commit hook to run all unit tests
# This ensures all tests pass before allowing a commit
# Exits with non-zero status if any test fails

set -e  # Exit immediately if a command exits with non-zero status

echo "Running unit tests..."

# Run pytest with the tests directory
# -v: verbose output
# --tb=short: shorter traceback format for cleaner output
uv run pytest tests/ -v --tb=short

echo "✓ All tests passed!"
