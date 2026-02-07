#!/bin/bash

set -e  # Exit on error

# Check if pyproject.toml.dev exists, if not create it from pyproject.toml
if [ ! -f "pyproject.toml.dev" ]; then
    echo "Creating pyproject.toml.dev from pyproject.toml..."
    cp pyproject.toml pyproject.toml.dev
fi

echo "Reading current version from pyproject.toml.dev..."
CURRENT_VERSION=$(grep 'version = ' pyproject.toml.dev | sed -E 's/version = "(.*)"/\1/')
echo "Current version: $CURRENT_VERSION"

# Parse version to increment dev number
if [[ $CURRENT_VERSION =~ ^(.+)\.dev([0-9]+)$ ]]; then
    # Already has .devN, increment N
    BASE_VERSION="${BASH_REMATCH[1]}"
    DEV_NUM="${BASH_REMATCH[2]}"
    NEW_DEV_NUM=$((DEV_NUM + 1))
    NEW_VERSION="${BASE_VERSION}.dev${NEW_DEV_NUM}"
elif [[ $CURRENT_VERSION =~ ^(.+)$ ]]; then
    # No .devN, add .dev1
    NEW_VERSION="${CURRENT_VERSION}.dev1"
else
    echo "Error: Could not parse version"
    exit 1
fi

echo "Updating version to: $NEW_VERSION in pyproject.toml.dev"
sed -i.bak "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml.dev
rm pyproject.toml.dev.bak

echo "Exporting environment variables from .env file..."
export $(grep -v '^#' .env | xargs)

echo "Cleaning old distributions..."
rm -rf dist/

# Temporarily use the dev config for building
echo "Building package with dev configuration..."
cp pyproject.toml pyproject.toml.backup
cp pyproject.toml.dev pyproject.toml

uv build > /dev/null 2>&1

# Restore original pyproject.toml
cp pyproject.toml.backup pyproject.toml
rm pyproject.toml.backup

echo "Uploading to Test PyPI..."
uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose

echo "Successfully published version $NEW_VERSION to Test PyPI"
echo "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ metrics-toolbox==$NEW_VERSION"
