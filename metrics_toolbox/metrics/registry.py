"""Metric registry for available metrics.

This module contains the METRIC_REGISTRY which maps metric type strings to their
corresponding class implementations.

To add new metrics, update the METRIC_REGISTRY with the new metric type and class name.
"""

from .enums import MetricEnum
from .roc_auc_binary import RocAucBinary
from .roc_auc_class import RocAucClass
from .roc_auc_macro import RocAucMacro
from .roc_auc_micro import RocAucMicro

METRIC_REGISTRY = {
    MetricEnum.ROC_AUC_BINARY: RocAucBinary,
    MetricEnum.ROC_AUC_MACRO: RocAucMacro,
    MetricEnum.ROC_AUC_MICRO: RocAucMicro,
    MetricEnum.ROC_AUC_CLASS: RocAucClass,
}
