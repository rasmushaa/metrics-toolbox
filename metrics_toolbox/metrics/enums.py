"""
This module contains the:
- MetricNameEnum: An enumeration of available metric names.
- MetricScopeEnum: An enumeration of metric scopes (binary, micro, macro, class).

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum


class MetricNameEnum(Enum):
    ROC_AUC_BINARY = "roc_auc_binary"
    ROC_AUC_MACRO = "roc_auc_macro"
    ROC_AUC_MICRO = "roc_auc_micro"
    ROC_AUC_CLASS = "roc_auc_class"


class MetricScopeEnum(Enum):
    BINARY = "binary"
    MICRO = "micro"
    MACRO = "macro"
    CLASS = "class"
