# Metrics Toolbox (readme under dev...)

A flexible toolbox for evaluating machine learning models with customizable metrics and reducers.

[![Tests](https://github.com/rasmushaa/metrics-toolbox/actions/workflows/test.yaml/badge.svg)](https://github.com/rasmushaa/metrics-toolbox/actions/workflows/test.yaml)
[![Coverage](https://codecov.io/gh/rasmushaa/metrics-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/rasmushaa/metrics-toolbox)
[![Python Version](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://rasmushaa.github.io/metrics-toolbox/)

## Features

- **Flexible Metric Configuration**: Add and configure metrics using enums, strings, or dictionaries
- **Type-Safe**: Leverage enums for type safety while maintaining flexibility with string names
- **Multiple Reducers**: Track metrics over time with various aggregation strategies (mean, min, max, std, etc.)
- **Built-in Metrics**: Pre-configured metrics for propability, label, and regression tasks on target, micro, and macro scopes
- **Chainable Builder Pattern**: Intuitive API for constructing metric evaluators
- **Visualization Support**: Generate ROC curves and other visualizations

## Available Metrics

| Name | Figures | Settings |
|------|---------|----------|
| `accuracy`        | Confusion matrix  | confusion_normalization |
| `precision_micro` | -                 | - |
| `precision_macro` | -                 | - |
| `precision_target`| -                 | target_name |
| `roc_auc_micro`   | Traces            | - |
| `roc_auc_macro`   | Traces            | - |
| `roc_auc_target`  | Traces            | target_name |

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
from metrics_toolbox import EvaluatorBuilder
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train a model with predict, and predict_proba methods
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Build evaluator with multiple metrics and reducers
evaluator = (
    EvaluatorBuilder()
    .add_metric("roc_auc_target", target_name=1, reducers=["mean", "std"])
    .add_metric("accuracy", reducers=["mean", "std"])
).build()

# 3. Evaluate model directly using multiple folds
evaluator.add_model_evaluation(model, X_test, y_test)
evaluator.add_model_evaluation(model, X_train, y_train)

# 4. Get aggregated results, history, and plots
result = evaluator.get_results()
display(result)

# 5. View figures
display(result['figures']['roc_auc_curves'])
display(result['figures']['confusion_matrices'])
```

## Usage
To see examples how to
- Use the builder pattern, see [builder examples notebook](examples/builder.ipynb)
- Get help, see the [help notebook](examples/help.ipynb)
- Use cases for model evaluation, see the [usecases notebook](examples/usecases.ipynb)
- Custome model evaluaton <TODO>

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/rasmushaa/metrics-toolbox.git
cd metrics-toolbox

# Install dependencies with uv
uv pip install -e ".[dev]"

# Install pre-commit hooks
uv run pre-commit install
```

### Testing

```bash
# Run tests with coverage, coverage is included in toml
uv run pytest
```

### Code Quality

```bash
# Run pre-commit hooks on all files
uv run pre-commit run --all-files

# Run pre-commit on staged files only
uv run pre-commit run
```

### Deployment

The project uses automated CI/CD:
- **Continuous Testing**: `main` and `feature/**` branches trigger the matrix testing of supported Python version
- **Continuous Requirements** `main` and `feature/**` branches trigger a job verifying test pipeline contains all python version listed on `pyproject.toml` *classifiers*
- **Continuous Documentation**: Pushes to the `main` branch automatically updates the **mkdocs API** refence, and uses the Readme as the landing page.
- **PyPI Deployment**: Create and push a version tag to main to trigger automated deployment to PyPI

```bash
# Deploy new version to PyPI
git tag v0.1.0
git push origin v0.1.0
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
