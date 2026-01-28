from sklearn.metrics import auc, roc_curve

from .base_metric import Metric
from .enums import MetricNameEnum, MetricScopeEnum
from .results import MetricResult


class RocAucBinary(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.BINARY
    _requires_probs = True

    def compute(self, y_true, y_pred, classes=None):
        """Compute ROC AUC for binary classification.

        Returns
        -------
        MetricResult
            The computed ROC AUC metric result, including FPR and TPR curves in metadata.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        value = auc(fpr, tpr)
        return MetricResult(
            name=self.name,
            value=value,
            metadata={"fpr": fpr, "tpr": tpr},
            scope=self.scope,
        )
