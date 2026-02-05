import numpy as np
from sklearn.metrics import auc, roc_curve

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class RocAucTarget(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.TARGET
    _type = MetricTypeEnum.PROBS

    def __init__(self, target_name: str):
        """Initialize the ROC AUC metric for a specific class.

        Parameters
        ----------
        target_name : str
            The class/column for which to compute the ROC AUC in a one-vs-all fashion.
        """
        self.target_name = target_name

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
        """Compute the ROC AUC for a specific class in a multi-class setting.

        This equals to binary ROC AUC where the positive class is `target_name` and
        all other classes are considered negative in a one-vs-all fashion.

        Parameters
        ----------
        y_true : np.ndarray
            True class labels binarized in one-vs-all fashion.
        y_pred : np.ndarray
            Predicted probabilities for each class.
        column_names : list[str], optional
            List of class names from model.classes_.

        Returns
        -------
        MetricResult
            The computed ROC AUC metric result for the specified target, including
            false positive rates (fpr) and true positive rates (tpr) in metadata.
        """

        class_index = column_names.index(self.target_name)
        fpr, tpr, _ = roc_curve(y_true[:, class_index], y_pred[:, class_index])
        value = auc(fpr, tpr)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            options={"target_name": self.target_name},
        )
