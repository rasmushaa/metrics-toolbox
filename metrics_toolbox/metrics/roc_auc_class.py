from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .base_metric import Metric
from .enums import MetricNameEnum, MetricScopeEnum, MetricTypeEnum
from .results import MetricResult


class RocAucClass(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.CLASS
    _type = MetricTypeEnum.PROBS

    def __init__(self, class_name: str):
        """Initialize the ROC AUC metric for a specific class.

        Parameters
        ----------
        class_name : str
            The class for which to compute the ROC AUC in a one-vs-all fashion.
        """
        self.class_name = class_name

    @property
    def id(self) -> str:
        """Get the unique identifier for the metric.

        Returns
        -------
        str
            The unique identifier.
        """
        return f"{self.name.value}_{self.scope.value}_{self.class_name}"

    def compute(self, y_true, y_pred, classes) -> MetricResult:
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

        class_index = classes.index(self.class_name)
        fpr, tpr, _ = roc_curve(
            y_true_binarized[:, class_index], y_pred[:, class_index]
        )
        value = auc(fpr, tpr)

        return MetricResult(
            name=self.name,
            scope=self.scope,
            type=self.type,
            value=value,
            metadata={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            options={"class_name": self.class_name},
        )
