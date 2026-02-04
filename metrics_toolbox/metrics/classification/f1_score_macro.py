import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class F1ScoreMacro(Metric):
    _name = MetricNameEnum.F1_SCORE
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.MACRO

    def __init__(self):
        """Initialize F1 score metric for classification."""

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str] = None
    ) -> MetricResult:
        """Compute F1 score for label classification.

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
            The computed F1 score metric result.
        """

        value = 0.0
        for i in range(len(column_names)):
            tp_c = sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
            fn_c = sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
            fp_c = sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
            f1_c = (
                2 * tp_c / (2 * tp_c + fn_c + fp_c)
                if (2 * tp_c + fn_c + fp_c) > 0
                else 0.0
            )
            value += f1_c
        value /= len(column_names)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
        )
