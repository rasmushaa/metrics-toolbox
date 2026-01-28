import numpy as np
import pytest

from metrics_toolbox.metrics.enums import MetricNameEnum, MetricScopeEnum
from metrics_toolbox.metrics.roc_auc_binary import RocAucBinary
from metrics_toolbox.metrics.roc_auc_class import RocAucClass
from metrics_toolbox.metrics.roc_auc_macro import RocAucMacro
from metrics_toolbox.metrics.roc_auc_micro import RocAucMicro


def test_roc_auc_binary_compute():
    """Test the RocAucMetric for binary classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.35, 0.6, 0.7, 0.2, 0.55, 0.05])

    metric = RocAucBinary()
    result = metric.compute(y_true, y_pred)

    assert result.name == MetricNameEnum.ROC_AUC
    assert result.scope == MetricScopeEnum.BINARY
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

    metric = RocAucMacro()
    result = metric.compute(y_true, y_prob, classes=classes)

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

    metric = RocAucMicro()
    result = metric.compute(y_true, y_prob, classes=classes)

    assert result.name == MetricNameEnum.ROC_AUC
    assert result.scope == MetricScopeEnum.MICRO
    assert "fpr" in result.metadata
    assert "tpr" in result.metadata
    assert result.value == pytest.approx(0.9687, abs=0.0001)


def test_roc_auc_class_compute():
    """Test the RocAucMetric for multi-class classification, class-specific.

    The mock values have not been calculated manually, but the test ensures that the
    class-specific ROC AUC matches the binary ROC AUC for that class, which has been
    verified separately.
    """
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

    target_class = 1

    # Class 1 specific ROC AUC
    metric = RocAucClass()
    result = metric.compute(y_true, y_prob, classes=classes, class_name=target_class)

    # The Binary ROC AUC for class 1
    metric_binary = RocAucBinary()
    result_binary = metric_binary.compute(
        (y_true == target_class).astype(int), y_prob[:, target_class]
    )

    assert result.name == MetricNameEnum.ROC_AUC
    assert result.scope == MetricScopeEnum.CLASS
    assert result.class_name == target_class
    assert "fpr" in result.metadata
    assert "tpr" in result.metadata
    assert result.value == pytest.approx(
        result_binary.value, abs=0.0001
    ), "The class-specific ROC AUC should match the binary ROC AUC for that class."
