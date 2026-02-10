#!/bin/bash

set -e  # Exit on error

# Read the actual production version from pyproject.toml
echo "Reading production version from pyproject.toml..."
PROD_VERSION=$(grep 'version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/')
echo "Production version: $PROD_VERSION"

# Check if pyproject.toml.dev exists
if [ ! -f "pyproject.toml.dev" ]; then
    echo "Creating pyproject.toml.dev from pyproject.toml..."
    cp pyproject.toml pyproject.toml.dev
    CURRENT_VERSION="$PROD_VERSION"
else
    echo "Reading current version from pyproject.toml.dev..."
    CURRENT_VERSION=$(grep 'version = ' pyproject.toml.dev | sed -E 's/version = "(.*)"/\1/')
    echo "Current dev version: $CURRENT_VERSION"
fi

# Extract base version from current dev version (strip .devN if present)
if [[ $CURRENT_VERSION =~ ^(.+)\.dev([0-9]+)$ ]]; then
    DEV_BASE_VERSION="${BASH_REMATCH[1]}"
    DEV_NUM="${BASH_REMATCH[2]}"
else
    DEV_BASE_VERSION="$CURRENT_VERSION"
    DEV_NUM=0
fi

# Check if base versions match
if [ "$DEV_BASE_VERSION" != "$PROD_VERSION" ]; then
    echo "Base version mismatch detected:"
    echo "  Dev base: $DEV_BASE_VERSION"
    echo "  Prod version: $PROD_VERSION"
    echo "Resetting to ${PROD_VERSION}.dev1"
    NEW_VERSION="${PROD_VERSION}.dev1"
else
    # Base versions match, increment dev number
    NEW_DEV_NUM=$((DEV_NUM + 1))
    NEW_VERSION="${PROD_VERSION}.dev${NEW_DEV_NUM}"
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
uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/* --username "__token__" --password "$TEST_PYPI_PASSWORD" --verbose

echo "Successfully published version $NEW_VERSION to Test PyPI"
echo "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ metrics-toolbox==$NEW_VERSION"
