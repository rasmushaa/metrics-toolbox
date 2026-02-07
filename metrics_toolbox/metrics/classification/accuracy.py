import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class Accuracy(Metric):
    _name = MetricNameEnum.ACCURACY
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.MICRO

    def __init__(self, opt_confusion_normalization: str = "true"):
        """Initialize Accuracy metric.

        Parameters
        ----------
        opt_confusion_normalization : str, optional
            Normalization mode for the confusion matrix. Can be 'true', 'pred', 'all', or None for no normalization.
        """
        self.opt_confusion_normalization = opt_confusion_normalization

    @property
    def id(self) -> str:
        """Get the unique identifier for the metric."""
        return self.name.value.lower()

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute accuracy for label classification.

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
            The computed accuracy metric result.
        """

        # Transform 2D one-hot encoded arrays to 1D label arrays using column_names
        y_true_label = np.array([column_names[i] for i in y_true.argmax(axis=1)])
        y_pred_label = np.array([column_names[i] for i in y_pred.argmax(axis=1)])

        value = accuracy_score(y_true_label, y_pred_label)

        cm = confusion_matrix(
            y_true_label,
            y_pred_label,
            labels=column_names,
            normalize=self.opt_confusion_normalization,
        )

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={
                "confusion_matrix": cm,
                "opt_confusion_normalization": self.opt_confusion_normalization,
                "class_names": column_names,
            },
        )
