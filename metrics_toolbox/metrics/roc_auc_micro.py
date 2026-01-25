from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .base import Metric
from .enums import MetricNameEnum, MetricScopeEnum
from .results import MetricResult


class RocAucMicro(Metric):
    _name = MetricNameEnum.ROC_AUC_MICRO
    _scope = MetricScopeEnum.MICRO
    _requires_probs = True
    _requires_classes = True

    def compute(self, y_true, y_pred, classes):

        # Binarize labels in a one-vs-all fashion -> shape (n_samples, n_classes).
        # Classes of [A,B,C] and 3 rows -> [[1,0,0],[0,1,0],[0,0,1]]
        y_true_binarized = label_binarize(y_true, classes=classes)

        fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred.ravel())
        value = auc(fpr, tpr)

        return MetricResult(
            name=self.name,
            value=value,
            metadata={"fpr": fpr, "tpr": tpr},
            scope=self.scope,
        )
