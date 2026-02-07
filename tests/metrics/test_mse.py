import numpy as np
import pytest

from metrics_toolbox.encoding import toolbox_widen_series
from metrics_toolbox.metrics.enums import MetricNameEnum, MetricScopeEnum
from metrics_toolbox.metrics.regression.mse_macro import MSEMacro
from metrics_toolbox.metrics.regression.mse_target import MSETarget
from metrics_toolbox.metrics.regression.rmse_macro import RMSEMacro
from metrics_toolbox.metrics.regression.rmse_target import RMSETarget


def test_mse_target_compute():
    """Test the MSETarget metric for regression.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2026-02-05.
    """
    y_true = np.array([1.0, 2.0, 3.0, -3.0, -2.0, -1.0, 0.0])
    y_pred = np.array([2.0, 2.0, 3.0, -5.0, -2.0, -1.0, 0.0])

    # The MetricsToolbox requires 2D inputs for all inputs, even for singel target regression tasks
    y_true_2d = toolbox_widen_series(y_true)
    y_pred_2d = toolbox_widen_series(y_pred)

    metric = MSETarget(target_name="test_target", opt_metadata_series_length=3)
    result = metric.compute(y_true_2d, y_pred_2d, column_names=["test_target"])
    print(result)

    assert result.name == MetricNameEnum.MSE
    assert result.scope == MetricScopeEnum.TARGET
    assert metric.id == MetricNameEnum.MSE.value + "_test_target"
    assert result.value == pytest.approx(5 / 7, abs=0.0001)
    assert result.metadata["y_true"] == [1.0, -3.0, 0.0]
    assert result.metadata["y_pred"] == [2.0, -5.0, 0.0]
    assert result.metadata["error"] == [1.0, 4.0, 0.0]


def test_rmse_target_compute():
    """Test the RMSETarget metric for regression.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2026-02-05.
    """
    y_true = np.array([1.0, 2.0, 3.0, -3.0, -2.0, -1.0, 0.0])
    y_pred = np.array([2.0, 2.0, 3.0, -5.0, -2.0, -1.0, 0.0])

    # The MetricsToolbox requires 2D inputs for all inputs, even for singel target regression tasks
    y_true_2d = toolbox_widen_series(y_true)
    y_pred_2d = toolbox_widen_series(y_pred)

    metric = RMSETarget(target_name="test_target", opt_metadata_series_length=3)
    result = metric.compute(y_true_2d, y_pred_2d, column_names=["test_target"])
    print(result)

    assert result.name == MetricNameEnum.RMSE
    assert result.scope == MetricScopeEnum.TARGET
    assert metric.id == MetricNameEnum.RMSE.value + "_test_target"
    assert result.value == pytest.approx(
        np.sqrt(5 / 7), abs=0.0001
    )  # RMSE is the square root of MSE
    assert result.metadata["y_true"] == [1.0, -3.0, 0.0]
    assert result.metadata["y_pred"] == [2.0, -5.0, 0.0]
    assert result.metadata["error"] == np.sqrt([1.0, 4.0, 0.0]).tolist()


def test_mse_macro_compute():
    """Test the MSEMacro metric for regression.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2026-02-05.
    """
    y_true = np.array([[1.0, 2.0], [3.0, -3.0], [-2.0, -1.0], [0.0, 4.0]])

    y_pred = np.array([[2.0, 1.0], [3.0, -5.0], [-2.0, -1.0], [0.0, 4.0]])

    metric = MSEMacro()
    result = metric.compute(y_true, y_pred, column_names=[0, 1])
    print(result)

    assert result.name == MetricNameEnum.MSE
    assert result.scope == MetricScopeEnum.MACRO
    assert metric.id == MetricNameEnum.MSE.value + "_" + MetricScopeEnum.MACRO.value
    assert result.value == pytest.approx(
        ((1 / 4) + (5 / 4)) / 2, abs=0.0001
    )  # Mean of every column MSE


def test_rmse_macro_compute():
    """Test the RMSEMacro metric for regression.

    The mock values have been calculated manually, by Rasmus Haapaniemi on 2026-02-05.
    """
    y_true = np.array([[1.0, 2.0], [3.0, -3.0], [-2.0, -1.0], [0.0, 4.0]])

    y_pred = np.array([[2.0, 1.0], [3.0, -5.0], [-2.0, -1.0], [0.0, 4.0]])

    metric = RMSEMacro()
    result = metric.compute(y_true, y_pred, column_names=[0, 1])
    print(result)

    assert result.name == MetricNameEnum.RMSE
    assert result.scope == MetricScopeEnum.MACRO
    assert metric.id == MetricNameEnum.RMSE.value + "_" + MetricScopeEnum.MACRO.value
    assert result.value == pytest.approx(
        (np.sqrt(1 / 4) + np.sqrt(5 / 4)) / 2, abs=0.0001
    )  # RMSE Macro is the mean of every column RMSE, and RMSE is the square root of MSE
