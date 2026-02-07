# !/bin/bash

echo "Exporting environment variables from .env file..."
export $(grep -v '^#' .env | xargs)

echo "Building the package..."
uv build > /dev/null 2>&1

echo "Uploading to Test PyPI..."
uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*
