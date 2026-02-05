import numpy as np
import pytest

from metrics_toolbox.encoding import toolbox_binarize_labels
from metrics_toolbox.metrics.classification.recall_macro import RecallMacro
from metrics_toolbox.metrics.classification.recall_micro import RecallMicro
from metrics_toolbox.metrics.classification.recall_target import RecallTarget
from metrics_toolbox.metrics.enums import MetricNameEnum, MetricScopeEnum


def test_recall_target_compute():
    """Test the RecallTarget metric for binary classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    # The MetricsToolbox requires binarized inputs for all inputs
    y_true_bin = toolbox_binarize_labels(y_true, classes=[0, 1])
    y_pred_bin = toolbox_binarize_labels(y_pred, classes=[0, 1])

    metric = RecallTarget(target_name=1)
    result = metric.compute(y_true_bin, y_pred_bin, column_names=[0, 1])

    assert result.name == MetricNameEnum.RECALL
    assert result.scope == MetricScopeEnum.TARGET
    assert metric.id == MetricNameEnum.RECALL.value + "_1"
    assert result.value == pytest.approx(0.75, abs=0.0001)


def test_recall_macro_compute():
    """Test the RecallMacro metric for multi-class classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 1, 2, 2])

    # The MetricsToolbox requires binarized inputs for all inputs
    y_true_bin = toolbox_binarize_labels(y_true, classes=[0, 1, 2])
    y_pred_bin = toolbox_binarize_labels(y_pred, classes=[0, 1, 2])

    metric = RecallMacro()
    result = metric.compute(y_true_bin, y_pred_bin, column_names=[0, 1, 2])

    assert result.name == MetricNameEnum.RECALL
    assert result.scope == MetricScopeEnum.MACRO
    assert metric.id == MetricNameEnum.RECALL.value + "_" + MetricScopeEnum.MACRO.value
    assert result.value == pytest.approx(0.8055, abs=0.0001)


def test_recall_micro_compute():
    """Test the RecallMicro metric for multi-class classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    # Test data as numpy arrays
    y_true = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 1, 2, 2])

    # The MetricsToolbox requires binarized inputs for all inputs
    y_true_bin = toolbox_binarize_labels(y_true, classes=[0, 1, 2])
    y_pred_bin = toolbox_binarize_labels(y_pred, classes=[0, 1, 2])

    metric = RecallMicro()
    result = metric.compute(y_true_bin, y_pred_bin, column_names=[0, 1, 2])

    assert result.name == MetricNameEnum.RECALL
    assert result.scope == MetricScopeEnum.MICRO
    assert metric.id == MetricNameEnum.RECALL.value + "_" + MetricScopeEnum.MICRO.value
    assert result.value == pytest.approx(0.75, abs=0.0001)
