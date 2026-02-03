import numpy as np
from sklearn.metrics import precision_score

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

        # Transform 2D one-hot encoded arrays to 1D label arrays using column_names
        y_true_label = np.array([column_names[i] for i in y_true.argmax(axis=1)])
        y_pred_label = np.array([column_names[i] for i in y_pred.argmax(axis=1)])

        value = precision_score(
            y_true_label,
            y_pred_label,
            average="macro",
            zero_division=0,
        )

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
        )
