from sklearn.metrics import auc, roc_curve

from .base_metric import Metric
from .enums import MetricNameEnum, MetricScopeEnum, MetricTypeEnum
from .results import MetricResult


class RocAucBinary(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.BINARY
    _type = MetricTypeEnum.PROBS

    def __init__(self):
        pass

    def compute(self, y_true, y_pred) -> MetricResult:
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
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        )
