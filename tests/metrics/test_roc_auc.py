import numpy as np
import pytest

from metrics_toolbox.encoding import toolbox_binarize_labels, toolbox_binarize_probs
from metrics_toolbox.metrics.enums import MetricNameEnum, MetricScopeEnum
from metrics_toolbox.metrics.prob.roc_auc_macro import RocAucMacro
from metrics_toolbox.metrics.prob.roc_auc_micro import RocAucMicro
from metrics_toolbox.metrics.prob.roc_auc_target import RocAucTarget


def test_roc_auc_target_compute():
    """Test the RocAucMetric for binary classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.35, 0.6, 0.7, 0.2, 0.55, 0.05])

    # The MetricsToolbox requires binarized inputs for all inputs
    y_true_bin = toolbox_binarize_labels(y_true, classes=[0, 1])
    y_pred_bin = toolbox_binarize_probs(y_pred)
    print(y_true_bin)
    print(y_pred_bin)

    metric = RocAucTarget(target_name=1)
    result = metric.compute(y_true_bin, y_pred_bin, column_names=[0, 1])

    assert result.name == MetricNameEnum.ROC_AUC
    assert result.scope == MetricScopeEnum.TARGET
    assert (
        metric.id
        == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.TARGET.value + "_1"
    )
    assert "fpr" in result.metadata
    assert "tpr" in result.metadata
    assert result.value == pytest.approx(0.9166, abs=0.0001)


def test_roc_auc_macro_compute():
    """Test the RocAucMetric for multi-class classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_prob = np.array(
        [
            [0.70, 0.20, 0.10],  # true 0
            [0.10, 0.70, 0.20],  # true 1
            [0.60, 0.30, 0.10],  # true 1 (bad prediction)
            [0.20, 0.60, 0.20],  # true 1
            [0.05, 0.20, 0.75],  # true 2
            [0.10, 0.40, 0.50],  # true 2
            [0.05, 0.30, 0.65],  # true 2
            [0.05, 0.10, 0.85],  # true 2
        ]
    )
    classes = [0, 1, 2]

    y_true_bin = toolbox_binarize_labels(y_true, classes=classes)

    metric = RocAucMacro()
    result = metric.compute(y_true_bin, y_prob, column_names=classes)

    assert result.name == MetricNameEnum.ROC_AUC
    assert result.scope == MetricScopeEnum.MACRO
    assert "fpr" in result.metadata
    assert "tpr" in result.metadata
    assert result.value == pytest.approx(0.9666, abs=0.0001)


def test_roc_auc_micro_compute():
    """Test the RocAucMetric for multi-class classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_prob = np.array(
        [
            [0.70, 0.20, 0.10],  # true 0
            [0.10, 0.70, 0.20],  # true 1
            [0.60, 0.30, 0.10],  # true 1 (bad prediction)
            [0.20, 0.60, 0.20],  # true 1
            [0.05, 0.20, 0.75],  # true 2
            [0.10, 0.40, 0.50],  # true 2
            [0.05, 0.30, 0.65],  # true 2
            [0.05, 0.10, 0.85],  # true 2
        ]
    )
    classes = [0, 1, 2]

    y_true_bin = toolbox_binarize_labels(y_true, classes=classes)

    metric = RocAucMicro()
    result = metric.compute(y_true_bin, y_prob, column_names=classes)

    assert result.name == MetricNameEnum.ROC_AUC
    assert result.scope == MetricScopeEnum.MICRO
    assert "fpr" in result.metadata
    assert "tpr" in result.metadata
    assert result.value == pytest.approx(0.9687, abs=0.0001)
