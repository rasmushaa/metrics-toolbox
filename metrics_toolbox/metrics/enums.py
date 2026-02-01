"""
This module contains the:

- **MetricNameEnum:** Available metric names.

- **MetricScopeEnum:** Metric scopes (micro, macro, class).

- **MetricTypeEnum:** The type of data the metric operates on (probabilities or labels).

The metrics and their scopes are separated to allow flexible combinations,
and easily identify the metric name and scope individually in the codebase.
"""

from enum import Enum


class MetricNameEnum(Enum):
    """Available metric function names."""

    ROC_AUC = "roc_auc"
    PRECISION = "precision"
    ACCURACY = "accuracy"
    RECALL = "recall"
    F1_SCORE = "f1_score"


class MetricScopeEnum(Enum):
    """Available metric scopes. (micro, macro, target)

    Attributes
    ----------
    MICRO : str
        Micro-averaged metric scope
    MACRO : str
        Macro-averaged metric scope
    TARGET : str
        Class/Column/Signal scope. This is a generic term for "per class" metrics
        for classification and "per column" metrics for regression.
        Also the binary classification metrics with two classes fall into this category,
        and the target name would be the positive class.
    """

    MICRO = "micro"
    MACRO = "macro"
    TARGET = "target"


class MetricTypeEnum(Enum):
    """Available metric types. (probs vs labels, etc.)

    Attributes
    ----------
    PROBS : str
        Metric operates on predicted probabilities 0-1.
    LABELS : str
        Metric operates on predicted class labels, string or int.
    """

    PROBS = "probs"
    LABELS = "labels"
