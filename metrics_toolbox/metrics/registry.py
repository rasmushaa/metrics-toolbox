"""
This module contains the:

- **MetricEnum:** The main end user enumeration for metrics.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum

from .classification.accuracy import Accuracy
from .classification.precision_macro import PrecisionMacro
from .classification.precision_micro import PrecisionMicro
from .classification.precision_target import PrecisionTarget
from .probability.roc_auc_macro import RocAucMacro
from .probability.roc_auc_micro import RocAucMicro
from .probability.roc_auc_target import RocAucTarget


class MetricEnum(Enum):
    """Enumeration of available metrics."""

    ACCURACY = Accuracy

    PRECISION_MICRO = PrecisionMicro
    PRECISION_MACRO = PrecisionMacro
    PRECISION_TARGET = PrecisionTarget

    ROC_AUC_MICRO = RocAucMicro
    ROC_AUC_MACRO = RocAucMacro
    ROC_AUC_TARGET = RocAucTarget
