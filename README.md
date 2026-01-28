# Metrics Toolbox (readme under dev...)

A flexible toolbox for evaluating machine learning models with customizable metrics and reducers.

[![Tests](https://github.com/rasmushaa/metrics-toolbox/actions/workflows/test.yaml/badge.svg)](https://github.com/rasmushaa/metrics-toolbox/actions/workflows/test.yaml)
[![Coverage](https://codecov.io/gh/rasmushaa/metrics-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/rasmushaa/metrics-toolbox)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://rasmushaa.github.io/metrics-toolbox/)

## Features

- **Flexible Metric Configuration**: Add and configure metrics using enums, strings, or dictionaries
- **Multiple Reducers**: Track metrics over time with various aggregation strategies (mean, min, max, std, etc.)
- **Built-in Metrics**: Pre-configured ROC AUC metrics (binary, macro, micro, per-class)
- **Chainable Builder Pattern**: Intuitive API for constructing metric evaluators
- **Visualization Support**: Generate ROC curves and other visualizations
- **Type-Safe**: Leverage enums for type safety while maintaining flexibility with string names

## Installation

```bash
pip install metrics-toolbox
```

## Quick Start

```python
from metrics_toolbox.builder import EvaluatorBuilder
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_classes=3, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# Build an evaluator
evaluator = (
    EvaluatorBuilder()
    .add_metric("roc_auc_binary")
    .add_metric("roc_auc_macro", reducers=["mean", "min", "max"])
    .add_metric("roc_auc_class", reducers=["latest"], class_name=1)
    .build()
)

# Evaluate your model
evaluator.update_model_evaluation(model, X_test, y_test)

# Get results
results = evaluator.get_results()
print(results)

# Visualize
evaluator.plot_auc_curves(X_test, y_test)
```

## Usage

### Building an Evaluator

**Using enums:**
```python
from metrics_toolbox.builder import EvaluatorBuilder
from metrics_toolbox.metrics.enums import MetricNameEnum
from metrics_toolbox.reducers.enums import ReducerEnum

evaluator = (
    EvaluatorBuilder()
    .add_metric(MetricNameEnum.ROC_AUC_BINARY)
    .add_metric(
        MetricNameEnum.ROC_AUC_MACRO,
        reducers=[ReducerEnum.MEAN, ReducerEnum.STD]
    )
    .build()
)
```

**Using strings:**
```python
evaluator = (
    EvaluatorBuilder()
    .add_metric("roc_auc_binary")
    .add_metric("roc_auc_macro", reducers=["mean", "std"])
    .build()
)
```

**From configuration dictionary:**
```python
config = {
    "metrics": [
        {"name": "roc_auc_binary"},
        {"name": "roc_auc_macro", "reducers": ["mean", "min"]},
        {"name": "roc_auc_class", "reducers": ["std"], "class_name": "A"}
    ]
}

evaluator = EvaluatorBuilder().from_dict(config).build()
```

### Tracking Metrics Over Time

```python
# Update evaluator multiple times (e.g., during training epochs)
for epoch in range(10):
    model.fit(X_train, y_train)
    evaluator.update_model_evaluation(model, X_val, y_val)

# Get aggregated results using configured reducers
results = evaluator.get_results()
```

## Available Metrics

- `roc_auc_binary` - ROC AUC for binary classification
- `roc_auc_macro` - Macro-averaged ROC AUC for multiclass
- `roc_auc_micro` - Micro-averaged ROC AUC for multiclass
- `roc_auc_class` - Per-class ROC AUC (specify `class_name`)

## Available Reducers

- `latest` - Most recent value
- `mean` - Average across all updates
- `min` - Minimum value
- `max` - Maximum value
- `std` - Standard deviation
- `minmax` - Difference between min and max

## Requirements

- Python >= 3.11
- NumPy >= 2.4.1
- scikit-learn >= 1.8.0
- matplotlib >= 3.10.8

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/metrics-toolbox.git
cd metrics-toolbox

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run pre-commit hooks
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
