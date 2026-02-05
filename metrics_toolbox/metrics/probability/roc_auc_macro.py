import numpy as np
from sklearn.metrics import auc, roc_curve

from metrics_toolbox.metrics.base_metric import Metric
from metrics_toolbox.metrics.enums import (
    MetricNameEnum,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.results import MetricResult


class RocAucMacro(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.MACRO
    _type = MetricTypeEnum.PROBS

    def __init__(self):
        pass

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """Compute macro-averaged ROC AUC for multi-class classification.

        Parameters
        ----------
        y_true : np.ndarray
            True class labels binarized in one-vs-all fashion.
        y_pred : np.ndarray
            Predicted probabilities for each class.
        column_names : list[str]
            List of class names from model.classes_.

        Returns
        -------
        MetricResult
            The computed macro-averaged ROC AUC value,
            including averaged FPR and TPR curves in metadata.
        """
        aucs = []
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        # Iterate over each class in binary fashion
        for i in range(len(column_names)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            value = auc(fpr, tpr)
            aucs.append(value)

            # Interpolate TPR at common FPR points
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        # Average AUC, and TPRs over all classes
        macro_auc = sum(aucs) / len(aucs)
        mean_tpr /= len(column_names)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=macro_auc,
            metadata={"fpr": all_fpr.tolist(), "tpr": mean_tpr.tolist()},
        )
