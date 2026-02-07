# Metrics Toolbox

Configurable ML evaluation toolkit with built-in cross-validation and metric aggregation.

[![Tests](https://github.com/rasmushaa/metrics-toolbox/actions/workflows/test.yaml/badge.svg)](https://github.com/rasmushaa/metrics-toolbox/actions/workflows/test.yaml)
[![Coverage](https://codecov.io/gh/rasmushaa/metrics-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/rasmushaa/metrics-toolbox)
[![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://rasmushaa.github.io/metrics-toolbox/)

## Features

- **Flexible Metric Configuration**: Add and configure metrics using enums, strings, or dictionaries
- **Type-Safe**: Leverage enums for type safety while maintaining flexibility with string names
- **Multiple Reducers**: Track metrics over time with various aggregation strategies (mean, min, max, std, etc.)
- **Built-in Metrics**: Pre-configured metrics for probability, label, and regression tasks on target, micro, and macro scopes
- **Chainable Builder Pattern**: Intuitive API for constructing metric evaluators
- **Visualization Support**: Generate ROC curves and other visualizations

## Available Classification Metrics
| Name | Figures | Settings |
|------|---------|----------|
| `accuracy`        | Confusion matrix  | opt_confusion_normalization |
| `precision_micro` | -                 | - |
| `precision_macro` | -                 | - |
| `precision_target`| -                 | target_name |
| `recall_micro`    | -                 | - |
| `recall_macro`    | -                 | - |
| `recall_target`   | -                 | target_name |
| `f1_score_micro`  | -                 | - |
| `f1_score_macro`  | -                 | - |
| `f1_score_target` | -                 | target_name |

## Available Probability Metrics
| Name | Figures | Settings |
|------|---------|----------|
| `roc_auc_micro`   | Traces            | - |
| `roc_auc_macro`   | Traces            | - |
| `roc_auc_target`  | Traces            | target_name |

## Available Regression Metrics
| Name | Figures | Settings |
|------|---------|----------|
| `mse_target`      | True, Pred, Error | target_name, opt_metadata_series_length |
| `mse_macro`       | -                 | - |
| `rmse_target`     | True, Pred, Error | target_name, opt_metadata_series_length |
| `rmse_macro`      | -                 | - |

## Available Reducers

| Name | Explanation |
|------|-------------|
| `latest` | Returns the most recent metric value |
| `mean` | Calculates the average of all metric values |
| `std` | Computes the standard deviation of metric values |
| `max` | Returns the maximum metric value |
| `min` | Returns the minimum metric value |
| `minmax` | Returns both minimum and maximum metric values |


## Installation

```bash
pip install metrics-toolbox
```

## Quick Start

```python
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from metrics_toolbox import EvaluatorBuilder
import numpy as np

# 1. Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a model
model = RandomForestClassifier(n_estimators=2, random_state=42, max_depth=3)
model.fit(X_train, y_train)

# 3. Build evaluator with multiple metrics, you can mix and match classification and probabilistic metrics
evaluator = (
    EvaluatorBuilder()
    .add_metric("roc_auc_target", target_name=1, reducers=["mean", "std"])
    .add_metric("accuracy", reducers=["mean", "std"])
    .add_metric("precision_target", target_name=1)
    .add_metric("recall_target", target_name=1)
    .add_metric("f1_score_target", target_name=1, reducers=["mean", "minmax"])
).build()

# 4. Evaluate model directly
evaluator.add_model_evaluation(model, X_test, y_test)

# 5. Add another evaluation on training set for comparison
evaluator.add_model_evaluation(model, X_train, y_train)

# 6. Get results
result = evaluator.get_results()
display(result['values'])
display(result['steps'])
display(result['figures'])

# 7. View figures
display(result['figures']['roc_auc_curves'])
display(result['figures']['confusion_matrices'])
```

## Usage
To see examples how to:
- Get help, see the [help notebook](https://github.com/rasmushaa/metrics-toolbox/blob/main/examples/help.ipynb)
- Use the builder pattern, see the [builder examples notebook](https://github.com/rasmushaa/metrics-toolbox/blob/main/examples/builder.ipynb)
- Binary classification model evaluation, see the [binary model notebook](https://github.com/rasmushaa/metrics-toolbox/blob/main/examples/binary_classification.ipynb)
- Multiclass classification model evaluation, see the [multiclass model notebook](https://github.com/rasmushaa/metrics-toolbox/blob/main/examples/mutliclass_classification.ipynb)
- Multivariate regression model evaluation, see the [regression model notebook](https://github.com/rasmushaa/metrics-toolbox/blob/main/examples/regression.ipynb)
- Custom Evaluator for custom model, see the [custom notebook](https://github.com/rasmushaa/metrics-toolbox/blob/main/examples/custom.ipynb)

## Development

### Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/rasmushaa/metrics-toolbox.git
cd metrics-toolbox

# Install in editable mode with development dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
uv run pre-commit install
```

### Testing

Run the test suite with coverage reporting:

```bash
uv run pytest
```

Coverage configuration is specified in `pyproject.toml`.
Also the `examples` are runned automatically as tests,
to keep those update according to the latest changes.

### Code Quality

The project uses pre-commit hooks to maintain code quality:

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run hooks on staged files only
uv run pre-commit run
```

### Test Deployment

Before publishing to the main PyPI repository, you can test the deployment process using TestPyPI:

```bash
bash scripts/publish_to_test_pypi.sh
```

This script will:
1. Create/update `pyproject.toml.dev` with auto-incremented version (e.g., `0.1.0.dev1`, `0.1.0.dev2`, etc.)
2. Build the package distribution files using the dev version
3. Load credentials from `.env` file (PAT and twine username)
4. Upload to TestPyPI (https://test.pypi.org/)
5. Keep your main `pyproject.toml` unchanged

The script automatically increments the `.dev<N>` suffix on each run, allowing unlimited test uploads without manual version management.

To install from TestPyPI for testing:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ metrics-toolbox
```

### Deployment

The project uses automated CI/CD workflows:

- **Continuous Testing**: Matrix testing across supported Python versions on `main` and `feature/**` branches
- **Requirements Validation**: Ensures test pipeline covers all Python versions listed in `pyproject.toml` classifiers
- **Documentation**: Automatically updates MkDocs API reference and deploys documentation on pushes to `main`
- **PyPI Publishing**: Automated deployment triggered by version tags

To release a new version:

1. Update the version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create a pull request to `main`
4. Add the `release` label to the pull request
5. Merge the pull request

The automated workflow will:
- Create and push a version tag (e.g., `v0.1.0`)
- Trigger the PyPI publishing workflow
- Build and publish the package to PyPI


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
