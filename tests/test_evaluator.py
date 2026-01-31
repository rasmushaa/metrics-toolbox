from metrics_toolbox.evaluator import MetricEvaluator
from metrics_toolbox.metrics.base_metric import (
    Metric,
    MetricResult,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.enums import MetricNameEnum
from metrics_toolbox.reducers.registry import ReducerEnum
from metrics_toolbox.spec import MetricSpec


# ------------------------- Configurable Mock Metrics -------------------------
class ConfigurableMockMetric(Metric):
    """A mock metric that returns different values on successive calls.

    This allows testing reducers by calling the evaluator multiple times
    with different metric values each time.

    Parameters
    ----------
    values : list[float]
        A sequence of values to return on successive compute() calls.
        If compute() is called more times than values provided,
        the last value will be repeated.

    Example
    -------
    >>> metric = ConfigurableMockMetric([0.5, 0.7, 0.9])
    >>> metric.compute(...)  # returns 0.5
    >>> metric.compute(...)  # returns 0.7
    >>> metric.compute(...)  # returns 0.9
    >>> metric.compute(...)  # returns 0.9 (repeats last)
    """

    def __init__(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        self._values = list(values)
        self._call_count = 0

    def compute(self, y_true, y_pred, **kwargs):
        # Get the value for this call
        value = self._values[min(self._call_count, len(self._values) - 1)]
        self._call_count += 1

        return MetricResult(
            name=self.name,
            type=self.type,
            value=value,
            metadata={"fpr": [], "tpr": []},
            scope=self.scope,
        )

    def reset(self):
        """Reset the call counter to start from the beginning."""
        self._call_count = 0


class ConfigurableMockMetricBinary(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.BINARY
    _type = MetricTypeEnum.PROBS


class ConfigurableMockMetricMacro(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.MACRO
    _type = MetricTypeEnum.PROBS


def test_evaluator_add_prob_evaluation():
    """Test that MetricEvaluator correctly updates Probabilistic metrics based on
    requirements."""
    evaluator = MetricEvaluator(
        metric_specs=[
            MetricSpec(
                ConfigurableMockMetricBinary([1, 2, 3]),
            ),
            MetricSpec(
                ConfigurableMockMetricMacro([1, 2, 3]),
                reducers=(ReducerEnum.MEAN, ReducerEnum.MIN),
            ),
        ]
    )

    # Mock data (Does not matter for mock metrics)
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1.0, 0, 0]

    # Update evaluation for all 3 mock values
    evaluator.add_prob_evaluation(y_true, y_pred, classes=[0, 1])
    evaluator.add_prob_evaluation(y_true, y_pred, classes=[0, 1])
    evaluator.add_prob_evaluation(y_true, y_pred, classes=[0, 1])

    # Get results
    results = evaluator.get_results()
    print(results)

    # Reducers should have been applied correctly
    assert results["values"]["roc_auc_binary_latest"] == 3
    assert results["values"]["roc_auc_macro_mean"] == 2
    assert results["values"]["roc_auc_macro_min"] == 1

    # Steps should be tracked correctly
    assert results["steps"]["roc_auc_binary_steps"] == [1, 2, 3]
    assert results["steps"]["roc_auc_macro_steps"] == [1, 2, 3]

    # There should be plots
    assert "roc_auc_curves" in results["figures"]
