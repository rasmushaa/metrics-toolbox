import numpy as np

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class MSETarget(Metric):
    _name = MetricNameEnum.MSE
    _scope = MetricScopeEnum.TARGET
    _type = MetricTypeEnum.SCORES

    def __init__(self, target_name: str, metadata_series_length: int = 1000):
        """Initialize the Mean Squared Error metric for a specific class.

        Parameters
        ----------
        target_name : str
            The class/column for which to compute the Mean Squared Error.
        metadata_series_length : int, optional
            The length to which the original series values and predicted series values
            will be down sampled in the metadata of the MetricResult.
        """
        self.target_name = target_name
        self.metadata_series_length = metadata_series_length

    @property
    def id(self) -> str:
        """Get the unique identifier for the metric.

        Returns
        -------
        str
            The unique identifier.
        """
        return f"{self.name.value}_{self.target_name}"

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute the Mean Squared Error for a specific column in a multi domain
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
            The computed Mean Squared Error metric result for the specified target, including
            false down sampled original series values and predicted series values in metadata.
        """

        class_index = column_names.index(self.target_name)

        # Compute the Mean Squared Error for the specified target column
        mse_array = (y_true[:, class_index] - y_pred[:, class_index]) ** 2
        value = mse_array.mean()

        # Down sample the original series values and predicted series values for metadata
        if len(y_true) > self.metadata_series_length:
            indices = np.linspace(
                0, len(y_true) - 1, self.metadata_series_length, dtype=int
            )
            y_true_sampled = y_true[indices, class_index]
            y_pred_sampled = y_pred[indices, class_index]
            mse_array_sampled = mse_array[indices]
        else:
            y_true_sampled = y_true[:, class_index]
            y_pred_sampled = y_pred[:, class_index]
            mse_array_sampled = mse_array

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={
                "y_true": y_true_sampled.tolist(),
                "y_pred": y_pred_sampled.tolist(),
                "error": mse_array_sampled.tolist(),
                "indices": (
                    indices.tolist()
                    if len(y_true) > self.metadata_series_length
                    else None
                ),
            },
            options={"target_name": self.target_name},
        )
