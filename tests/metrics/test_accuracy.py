import numpy as np
import pytest

from metrics_toolbox.encoding import toolbox_binarize_labels
from metrics_toolbox.metrics.classification.accuracy import Accuracy
from metrics_toolbox.metrics.enums import MetricNameEnum, MetricScopeEnum


def test_accuracy_compute():
    """Test the Accuracy metric for label classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    y_true_bin = toolbox_binarize_labels(y_true, classes=[0, 1])
    y_pred_bin = toolbox_binarize_labels(y_pred, classes=[0, 1])
    print(y_true_bin)
    print(y_pred_bin)

    metric = Accuracy(confusion_normalization="true")
    result = metric.compute(y_true_bin, y_pred_bin, column_names=[0, 1])
    print(result)

    assert result.name == MetricNameEnum.ACCURACY
    assert result.scope == MetricScopeEnum.MICRO
    assert metric.id == MetricNameEnum.ACCURACY.value
    assert "confusion_matrix" in result.metadata
    assert result.value == pytest.approx(0.70, abs=0.01)
    assert result.metadata["confusion_matrix"].shape == (2, 2)
    expected_cm = np.array([[0.666, 0.333], [0.25, 0.75]])
    np.testing.assert_almost_equal(
        result.metadata["confusion_matrix"], expected_cm, decimal=2
    )


def test_accuracy_compute_no_normalization():
    """Test the Accuracy metric for label classification.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2025-11-23.
    """
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    y_true_bin = toolbox_binarize_labels(y_true, classes=[0, 1])
    y_pred_bin = toolbox_binarize_labels(y_pred, classes=[0, 1])
    print(y_true_bin)
    print(y_pred_bin)

    metric = Accuracy(confusion_normalization=None)
    result = metric.compute(y_true_bin, y_pred_bin, column_names=[0, 1])
    print(result)

    assert result.name == MetricNameEnum.ACCURACY
    assert result.scope == MetricScopeEnum.MICRO
    assert metric.id == MetricNameEnum.ACCURACY.value
    assert "confusion_matrix" in result.metadata
    assert result.value == pytest.approx(0.70, abs=0.01)
    assert result.metadata["confusion_matrix"].shape == (2, 2)
    expected_cm = np.array([[4, 2], [1, 3]])
    np.testing.assert_almost_equal(
        result.metadata["confusion_matrix"], expected_cm, decimal=2
    )
