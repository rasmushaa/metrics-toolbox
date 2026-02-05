import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class PrecisionMicro(Metric):
    _name = MetricNameEnum.PRECISION
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.MICRO

    def __init__(self):
        """Initialize Precision metric for classification."""

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

        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        tp = sum((y_pred_flat == 1) & (y_true_flat == 1))
        fp = sum((y_pred_flat == 1) & (y_true_flat == 0))

        value = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
        )
