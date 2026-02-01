import numpy as np
from sklearn.metrics import auc, roc_curve

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class RocAucMicro(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.MICRO
    _type = MetricTypeEnum.PROBS

    def __init__(self):
        pass

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str] = None
    ) -> MetricResult:
        """Compute the ROC AUC for micro-averaged multiclass classification.

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
            The computed ROC AUC metric result with FPR and TPR in metadata,
            including the tpr and fpr values for plotting the ROC curve.
        """

        fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        value = auc(fpr, tpr)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        )
