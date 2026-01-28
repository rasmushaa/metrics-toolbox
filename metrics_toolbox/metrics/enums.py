"""
This module contains the:

- **MetricEnum:** The main end user enumeration for metrics. (Combines name and scope)
- **MetricNameEnum:** Available metric names.
- **MetricScopeEnum:** Metric scopes (binary, micro, macro, class).

The metrics and their scopes are seperated to allow flexible combinations,
and easily identify the metric name and scope individually in the codebase.

The **MetricEnum** combines these two aspects for convenience,
and is the only enumeration that should be used by end users.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum


class MetricNameEnum(Enum):
    ROC_AUC = "roc_auc"


class MetricScopeEnum(Enum):
    BINARY = "binary"
    MICRO = "micro"
    MACRO = "macro"
    CLASS = "class"


class MetricEnum(Enum):
    ROC_AUC_BINARY = MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.BINARY.value
    ROC_AUC_MICRO = MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.MICRO.value
    ROC_AUC_MACRO = MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.MACRO.value
    ROC_AUC_CLASS = MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.CLASS.value
