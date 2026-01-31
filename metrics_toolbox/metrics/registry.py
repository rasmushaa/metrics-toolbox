"""
This module contains the:

- **MetricEnum:** The main end user enumeration for metrics.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum

from .roc_auc_binary import RocAucBinary
from .roc_auc_class import RocAucClass
from .roc_auc_macro import RocAucMacro
from .roc_auc_micro import RocAucMicro


class MetricEnum(Enum):
    ROC_AUC_BINARY = RocAucBinary
    ROC_AUC_MICRO = RocAucMicro
    ROC_AUC_MACRO = RocAucMacro
    ROC_AUC_CLASS = RocAucClass
