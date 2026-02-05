import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class F1ScoreTarget(Metric):
    _name = MetricNameEnum.F1_SCORE
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.TARGET

    def __init__(self, target_name: str):
        """Initialize F1 score metric for classification.

        Parameters
        ----------
        target_name : str
            Name of the target variable.
        """
        self.target_name = target_name

    @property
    def id(self) -> str:
        """Get the unique identifier for the metric.

        Returns
        -------
        str
            Unique identifier combining name, scope, and target name.
        """
        return self.name.value + "_" + str(self.target_name)

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute F1 score for label classification.

        Parameters
        ----------
        y_true : array-like of shape (n_samples, n_classes)
            True binary labels in one-hot encoded format.
        y_pred : array-like of shape (n_samples, n_classes)
            Predicted binary labels in one-hot encoded format.
        column_names : list[str]
            Class names corresponding to column indices.

        Returns
        -------
        MetricResult
            The computed F1 score metric result.
        """

        target_index = column_names.index(self.target_name)

        tp_c = sum((y_pred[:, target_index] == 1) & (y_true[:, target_index] == 1))
        fn_c = sum((y_pred[:, target_index] == 0) & (y_true[:, target_index] == 1))
        fp_c = sum((y_pred[:, target_index] == 1) & (y_true[:, target_index] == 0))

        f1_c = (
            2 * tp_c / (2 * tp_c + fn_c + fp_c) if (2 * tp_c + fn_c + fp_c) > 0 else 0.0
        )

        return MetricResult(
            name=self.name, scope=self.scope, type=self.type, value=f1_c
        )
