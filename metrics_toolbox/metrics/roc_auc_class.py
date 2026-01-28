from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .base_metric import Metric
from .enums import MetricNameEnum, MetricScopeEnum
from .results import MetricResult


class RocAucClass(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.CLASS
    _requires_probs = True
    _requires_classes = True

    def compute(self, y_true, y_pred, classes, class_name):
        """Compute the ROC AUC for a specific class in a multi-class setting.

        This equals to binary ROC AUC where the positive class is `class_name` and
        all other classes are considered negative in a one-vs-all fashion.

        Returns
        -------
        MetricResult
            The computed ROC AUC metric result for the specified class, including
            false positive rates (fpr) and true positive rates (tpr) in metadata.
        """

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
            metadata={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            scope=self.scope,
            class_name=class_name,
        )
