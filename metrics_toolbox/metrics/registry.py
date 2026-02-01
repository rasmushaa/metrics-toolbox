"""
This module contains the:

- **MetricEnum:** The main end user enumeration for metrics.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum

from .prob.roc_auc_macro import RocAucMacro
from .prob.roc_auc_micro import RocAucMicro
from .prob.roc_auc_target import RocAucTarget


class MetricEnum(Enum):
    """Enumeration of available metrics."""

    ROC_AUC_MICRO = RocAucMicro
    ROC_AUC_MACRO = RocAucMacro
    ROC_AUC_TARGET = RocAucTarget
