import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class MSEMacro(Metric):
    _name = MetricNameEnum.MSE
    _scope = MetricScopeEnum.MACRO
    _type = MetricTypeEnum.SCORES

    def __init__(self):
        """Initialize the Mean Squared Error metric for a all columns.

        Parameters
        ----------
        target_name : str
            The class/column for which to compute the Mean Squared Error.
        """

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute the Mean Squared Error for all columns in a multi domain setting, and
        return the average of the Mean Squared Errors for all columns as the final
        metric value.

        Parameters
        ----------
        y_true : np.ndarray
            True series values for the specified target column.
        y_pred : np.ndarray
            Predicted series values for the specified target column.
        column_names : list[str], optional
            List of column names from model.classes_.

        Returns
        -------
        MetricResult
            The computed Mean Squared Error metric result for the specified target.
        """

        # Compute the Mean Squared Error for the each column
        mse_array = np.mean((y_true - y_pred) ** 2, axis=0)
        value = mse_array.mean()

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
        )
