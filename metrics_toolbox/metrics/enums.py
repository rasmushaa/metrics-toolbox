"""
This module contains the:

- **MetricNameEnum:** Available metric names.

- **MetricScopeEnum:** Metric scopes (binary, micro, macro, class).

- **MetricTypeEnum:** The type of data the metric operates on (probabilities or labels).

The metrics and their scopes are seperated to allow flexible combinations,
and easily identify the metric name and scope individually in the codebase.
"""

from enum import Enum


class MetricNameEnum(Enum):
    ROC_AUC = "roc_auc"


class MetricScopeEnum(Enum):
    BINARY = "binary"
    MICRO = "micro"
    MACRO = "macro"
    CLASS = "class"


class MetricTypeEnum(Enum):
    PROBS = "probs"
    LABELS = "labels"
