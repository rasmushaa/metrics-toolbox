import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class RMSEMacro(Metric):
    _name = MetricNameEnum.RMSE
    _scope = MetricScopeEnum.MACRO
    _type = MetricTypeEnum.SCORES

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute the Root Mean Squared Error for a all columns in a multi domain
        setting, and then take the mean of the column-wise RMSE values to get the macro
        RMSE.

        Parameters
        ----------
        y_true : np.ndarray
            True series values for all target columns.
        y_pred : np.ndarray
            Predicted series values for all target columns.
        column_names : list[str], optional
            List of column names from model.classes_.

        Returns
        -------
        MetricResult
            The computed Root Mean Squared Error metric from the mean of column-wise RMSE values.
        """
        # Compute the Mean Squared Error for the each column
        mse_array = np.mean((y_true - y_pred) ** 2, axis=0)

        # Scale the MSE values to RMSE by taking the square root
        rmse_array = np.sqrt(mse_array)

        # Take the mean of the column-wise RMSE values to get the macro RMSE
        rmse_macro_value = rmse_array.mean()

        return MetricResult(
            name=self._name,
            scope=self._scope,
            type=self._type,
            value=rmse_macro_value,
        )
