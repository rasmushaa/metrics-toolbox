import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class RecallMacro(Metric):
    _name = MetricNameEnum.RECALL
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.MACRO

    def __init__(self):
        """Initialize Recall metric for classification."""

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str] = None
    ) -> MetricResult:
        """Compute recall for label classification.

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
            The computed recall metric result.
        """

        value = 0.0
        for i in range(len(column_names)):
            tp_c = sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
            fn_c = sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
            recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            value += recall_c
        value /= len(column_names)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
        )
