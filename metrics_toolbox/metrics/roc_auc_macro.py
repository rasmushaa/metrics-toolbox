import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .base_metric import Metric
from .enums import MetricNameEnum, MetricScopeEnum, MetricTypeEnum
from .results import MetricResult


class RocAucMacro(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.MACRO
    _type = MetricTypeEnum.PROBS

    def __init__(self):
        pass

    def compute(self, y_true, y_pred, classes) -> MetricResult:
        """Compute macro-averaged ROC AUC for multi-class classification.

        Returns
        -------
        MetricResult
            The computed macro-averaged ROC AUC metric result, including averaged FPR and TPR curves in metadata.
        """

        # Binarize labels in a one-vs-all fashion -> shape (n_samples, n_classes).
        # Classes of [A,B,C] and 3 rows -> [[1,0,0],[0,1,0],[0,0,1]]
        y_true_binarized = label_binarize(y_true, classes=classes)

        aucs = []
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        # Iterate over each class in binary fashion
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
            value = auc(fpr, tpr)
            aucs.append(value)

            # Interpolate TPR at common FPR points
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        # Average AUC, and TPRs over all classes
        macro_auc = sum(aucs) / len(aucs)
        mean_tpr /= len(classes)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=macro_auc,
            metadata={"fpr": all_fpr.tolist(), "tpr": mean_tpr.tolist()},
        )
