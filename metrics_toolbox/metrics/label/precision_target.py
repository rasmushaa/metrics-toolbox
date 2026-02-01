import numpy as np
from sklearn.metrics import confusion_matrix, precision_score

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class PrecisionTarget(Metric):
    _name = MetricNameEnum.PRECISION
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.TARGET

    def __init__(self, target_name: str, confusion_normalization: str = "true"):
        """Initialize Precision metric for binary classification.

        Parameters
        ----------
        target_name : str
            Name of the target variable.
        confusion_normalization : str, optional
            Normalization mode for the confusion matrix. Can be 'true', 'pred', 'all', or None for no normalization.
        """
        self.target_name = target_name
        self.confusion_normalization = confusion_normalization

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
            Class names corresponding to the binary labels.

        Returns
        -------
        MetricResult
            The computed precision metric result.
        """

        # Transform 2D one-hot encoded arrays to 1D label arrays
        y_true_label = y_true.argmax(axis=1)
        y_pred_label = y_pred.argmax(axis=1)

        value = precision_score(
            y_true_label,
            y_pred_label,
            labels=[self.target_name],
            average=None,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_true, y_pred, labels=column_names, normalize=self.confusion_normalization
        )

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={"confusion_matrix": cm},
        )
