from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from .base import Metric
from .results import MetricResult
from .enums import MetricNameEnum, MetricScopeEnum


class RocAucClass(Metric):
    _name = MetricNameEnum.ROC_AUC_CLASS
    _scope = MetricScopeEnum.CLASS
    _requires_probs = True

    def compute(self, y_true, y_pred, classes, class_name):

        # Binarize labels in a one-vs-all fashion -> shape (n_samples, n_classes).
        # Classes of [A,B,C] and 3 rows -> [[1,0,0],[0,1,0],[0,0,1]]
        y_true_binarized = label_binarize(y_true, classes=classes)

        class_index = classes.index(class_name)
        fpr, tpr, _ = roc_curve(
            y_true_binarized[:, class_index], y_pred[:, class_index]
        )
        value = auc(fpr, tpr)

        return MetricResult(
            name=self.name,
            value=value,
            metadata={"fpr": fpr, "tpr": tpr},
            scope=self.scope,
            class_name=class_name,
        )
