from sklearn.metrics import confusion_matrix, precision_score

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class Precision(Metric):
    _name = MetricNameEnum.PRECISION
    _type = MetricTypeEnum.LABELS
    _scope = MetricScopeEnum.TARGET

    def __init__(self, confusion_normalization: str = "true"):
        """Initialize Precision metric for binary classification.

        Parameters
        ----------
        confusion_normalization : str, optional
            Normalization mode for the confusion matrix. Can be 'true', 'pred', 'all', or None for no normalization.
        """
        self.confusion_normalization = confusion_normalization

    def compute(self, y_true, y_pred, classes: list[str] = None) -> MetricResult:
        """Compute precision for label classification.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels.
        classes : list[str], optional
            Labels to index the confusion matrix. If None, defaults to sorted unique labels in y_true and y_pred.

        Returns
        -------
        MetricResult
            The computed precision metric result.
        """
        value = precision_score(y_true, y_pred, average="binary")
        cm = confusion_matrix(
            y_true, y_pred, labels=classes, normalize=self.confusion_normalization
        )
        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={"confusion_matrix": cm},
        )
