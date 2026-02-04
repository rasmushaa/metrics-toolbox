import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class PrecisionMacro(Metric):
    _name = MetricNameEnum.PRECISION
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.MACRO

    def __init__(self):
        """Initialize Precision metric for binary classification."""

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str] = None
    ) -> MetricResult:
        """Compute precision for label classification.

        Parameters
        ----------
        y_true : array-like of shape (n_samples, n_classes)
            True binary labels in one-hot encoded format.
        y_pred : array-like of shape (n_samples, n_classes)
            Predicted binary labels in one-hot encoded format.
        column_names : list[str], optional
            Class names corresponding to column indices.

        Returns
        -------
        MetricResult
            The computed precision metric result.
        """

        value = 0.0
        for i in range(len(column_names)):
            tp_c = sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
            fp_c = sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
            precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            value += precision_c
        value /= len(column_names)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
        )
