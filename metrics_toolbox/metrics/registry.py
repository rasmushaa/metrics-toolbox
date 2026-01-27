"""Metric registry for available metrics.

This module contains the METRIC_REGISTRY which maps metric type strings to their
corresponding class implementations.

To add new metrics, update the METRIC_REGISTRY with the new metric type and class name.
"""

from .enums import MetricNameEnum
from .roc_auc_binary import RocAucBinary
from .roc_auc_class import RocAucClass
from .roc_auc_macro import RocAucMacro
from .roc_auc_micro import RocAucMicro

METRIC_REGISTRY = {
    MetricNameEnum.ROC_AUC_BINARY.value: RocAucBinary,
    MetricNameEnum.ROC_AUC_MACRO.value: RocAucMacro,
    MetricNameEnum.ROC_AUC_MICRO.value: RocAucMicro,
    MetricNameEnum.ROC_AUC_CLASS.value: RocAucClass,
}
