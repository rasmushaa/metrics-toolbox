from metrics_toolbox.evaluator import MetricEvaluator
from metrics_toolbox.metrics.base import Metric, MetricResult, MetricScopeEnum
from metrics_toolbox.metrics.enums import MetricNameEnum
from metrics_toolbox.reducers.enums import MetricReducerEnum
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
            value=value,
            metadata={"fpr": [], "tpr": []},
            scope=self.scope,
        )

    def reset(self):
        """Reset the call counter to start from the beginning."""
        self._call_count = 0


class ConfigurableMockMetricBinary(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC_BINARY
    _scope = MetricScopeEnum.BINARY
    _requires_probs = True


class ConfigurableMockMetricMacro(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC_MACRO
    _scope = MetricScopeEnum.MACRO
    _requires_probs = True


class ConfigurableMockMetricClass(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC_CLASS
    _scope = MetricScopeEnum.CLASS
    _requires_probs = True


# ------------------------- Mock Specs -------------------------
binary_metric_spec = MetricSpec(metric_cls=ConfigurableMockMetricBinary([1, 2, 3]))

macro_metric_spec_params = MetricSpec(
    metric_cls=ConfigurableMockMetricMacro([1, 2, 3]),
    reducers=(MetricReducerEnum.MEAN, MetricReducerEnum.MIN),
)

class_metric_spec = MetricSpec(
    metric_cls=ConfigurableMockMetricClass([1, 2, 3]), class_name="A"
)


def test_metric_evaluator_probabilities():
    """Test that MetricEvaluator correctly updates Probabilistic metrics based on
    requirements."""
    evaluator = MetricEvaluator(
        metric_specs=[
            binary_metric_spec,
            macro_metric_spec_params,
            class_metric_spec,
        ]
    )

    # Mock data (Does not matter for mock metrics)
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]

    # Update evaluation for all 3 mock values
    evaluator.update_probs_evaluation(y_true, y_pred, classes=[0, 1])
    evaluator.update_probs_evaluation(y_true, y_pred, classes=[0, 1])
    evaluator.update_probs_evaluation(y_true, y_pred, classes=[0, 1])
    print(evaluator)

    results = evaluator.results()
    print(results)

    # Check metric values
    assert results["reduced"]["roc_auc_binary_latest"] == 3
    assert results["reduced"]["roc_auc_macro_mean"] == 2.0
    assert results["reduced"]["roc_auc_macro_min"] == 1
    assert results["reduced"]["roc_auc_class_A_latest"] == 3

    # Check history values
    assert results["history"]["roc_auc_binary_steps"] == [1, 2, 3]
    assert results["history"]["roc_auc_macro_steps"] == [1, 2, 3]
    assert results["history"]["roc_auc_class_A_steps"] == [1, 2, 3]
