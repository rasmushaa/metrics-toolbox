from sklearn.metrics import roc_curve, auc
from .base import Metric
from .results import MetricResult
from .enums import MetricNameEnum, MetricScopeEnum


class RocAucBinary(Metric):
    _name = MetricNameEnum.ROC_AUC_BINARY
    _scope = MetricScopeEnum.BINARY
    _requires_probs = True

    def compute(self, y_true, y_pred, classes=None):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        value = auc(fpr, tpr)
        return MetricResult(
            name=self.name,
            value=value,
            metadata={"fpr": fpr, "tpr": tpr},
            scope=self.scope,
        )
