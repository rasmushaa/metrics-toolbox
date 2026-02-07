import numpy as np

from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.regression.mse_target import MSETarget
from metrics_toolbox.metrics.results import MetricResult


class RMSETarget(MSETarget):
    _name = MetricNameEnum.RMSE
    _scope = MetricScopeEnum.TARGET
    _type = MetricTypeEnum.SCORES

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute the Root Mean Squared Error for a specific column in a multi domain
        setting.

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
            The computed Root Mean Squared Error metric result for the specified target, including
            false down sampled original series values and predicted series values in metadata.
        """
        # Compute MSE using the parent class method
        mse_result = super().compute(y_true, y_pred, column_names)

        # Scale the MSE value to RMSE by taking the square root
        rmse_value = np.sqrt(mse_result.value)
        rmse_metadata = mse_result.metadata.copy()
        rmse_metadata["error"] = np.sqrt(rmse_metadata["error"]).tolist()
        rmse_metadata["y_true"] = mse_result.metadata["y_true"]
        rmse_metadata["y_pred"] = mse_result.metadata["y_pred"]
        rmse_metadata["indices"] = mse_result.metadata.get("indices", None)

        return MetricResult(
            name=self._name,
            scope=self._scope,
            type=self._type,
            value=rmse_value,
            metadata=rmse_metadata,
            options=mse_result.options,
        )
