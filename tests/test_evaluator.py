import numpy as np
import pytest

from metrics_toolbox.evaluator import MetricEvaluator
from metrics_toolbox.metrics.base_metric import (
    Metric,
    MetricResult,
    MetricScopeEnum,
    MetricTypeEnum,
)
from metrics_toolbox.metrics.classification.accuracy import Accuracy
from metrics_toolbox.metrics.classification.f1_score_target import F1ScoreTarget
from metrics_toolbox.metrics.classification.precision_target import PrecisionTarget
from metrics_toolbox.metrics.classification.recall_target import RecallTarget
from metrics_toolbox.metrics.enums import MetricNameEnum
from metrics_toolbox.metrics.probability.roc_auc_target import RocAucTarget
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
            metadata={
                "fpr": [],
                "tpr": [],
                "confusion_matrix": np.array([[0, 0], [0, 0]]),
            },
            scope=self.scope,
        )

    def reset(self):
        """Reset the call counter to start from the beginning."""
        self._call_count = 0


class ConfigurableMockMetricBinary(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.TARGET
    _type = MetricTypeEnum.PROBS


class ConfigurableMockMetricMacro(ConfigurableMockMetric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.MACRO
    _type = MetricTypeEnum.PROBS


class ConfigurableMockMetricLabel(ConfigurableMockMetric):
    _name = MetricNameEnum.ACCURACY
    _scope = MetricScopeEnum.TARGET
    _type = MetricTypeEnum.LABELS


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
    y_true = [[0], [1], [0], [1]]
    y_pred = [[0], [1.0], [0], [0]]

    # Update evaluation for all 3 mock values
    evaluator.add_prob_evaluation(y_true, y_pred, column_names=[1])
    evaluator.add_prob_evaluation(y_true, y_pred, column_names=[1])
    evaluator.add_prob_evaluation(y_true, y_pred, column_names=[1])

    # Get results
    results = evaluator.get_results()
    print(results)

    # Reducers should have been applied correctly
    assert results["values"]["roc_auc_target_latest"] == 3
    assert results["values"]["roc_auc_macro_mean"] == 2
    assert results["values"]["roc_auc_macro_min"] == 1

    # Steps should be tracked correctly
    assert results["steps"]["roc_auc_target_steps"] == [1, 2, 3]
    assert results["steps"]["roc_auc_macro_steps"] == [1, 2, 3]

    # There should be plots
    assert "roc_auc_curves" in results["figures"]


def test_evaluator_add_label_evaluation():
    """Test that MetricEvaluator correctly updates Label metrics based on
    requirements."""
    evaluator = MetricEvaluator(
        metric_specs=[
            MetricSpec(
                ConfigurableMockMetricLabel([0.8, 0.85, 0.9]),
            ),
        ]
    )

    # Mock data (Does not matter for mock metrics)
    y_true = [[0], [1], [0], [1]]
    y_pred = [[0], [1], [0], [0]]

    # Update evaluation for all 3 mock values
    evaluator.add_label_evaluation(y_true, y_pred, column_names=[1])
    evaluator.add_label_evaluation(y_true, y_pred, column_names=[1])
    evaluator.add_label_evaluation(y_true, y_pred, column_names=[1])

    # Get results
    results = evaluator.get_results()
    print(results)

    # Latest value should be the last one
    assert (
        results["values"]["accuracy_target_latest"] == 0.9
    )  # The mock metric does not have id override...

    # Steps should be tracked correctly
    assert results["steps"]["accuracy_target_steps"] == [0.8, 0.85, 0.9]


def test_evaluator_add_model_evaluation():
    """Test that MetricEvaluator correctly evaluates a model using both predict and
    predict_proba methods."""

    class MockModel:
        """A mock model that returns predefined predictions."""

        def __init__(self, label_predictions, prob_predictions):
            self.label_predictions = label_predictions
            self.prob_predictions = prob_predictions
            self.predict_call_count = 0
            self.predict_proba_call_count = 0

        def predict(self, X):
            """Return predefined label predictions."""
            result = self.label_predictions[self.predict_call_count]
            self.predict_call_count += 1
            return result

        def predict_proba(self, X):
            """Return predefined probability predictions."""
            result = self.prob_predictions[self.predict_proba_call_count]
            self.predict_proba_call_count += 1
            return result

        @property
        def classes_(self):
            return [0, 1]

    # Create mock model with 1 sets of predictions
    mock_model = MockModel(
        label_predictions=[np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])],
        prob_predictions=[
            np.array([0.1, 0.9, 0.4, 0.8, 0.35, 0.6, 0.7, 0.2, 0.55, 0.05])
        ],
    )

    evaluator = MetricEvaluator(
        metric_specs=[
            MetricSpec(
                RocAucTarget(target_name=1),
            ),
            MetricSpec(
                Accuracy(),
            ),
            MetricSpec(
                PrecisionTarget(target_name=1),
            ),
            MetricSpec(
                RecallTarget(target_name=1),
            ),
            MetricSpec(
                F1ScoreTarget(target_name=1),
            ),
        ]
    )

    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    X = np.zeros((10, 5))  # Mock feature data (not used by MockModel)

    evaluator.add_model_evaluation(mock_model, X, y_true)
    results = evaluator.get_results()
    print(results)

    assert results["values"]["accuracy_latest"] == pytest.approx(0.7, abs=0.0001)
    assert results["values"]["roc_auc_1_latest"] == pytest.approx(0.9166, abs=0.0001)
    assert "roc_auc_curves" in results["figures"]
    assert results["values"]["precision_1_latest"] == pytest.approx(0.6, abs=0.0001)
    assert results["values"]["recall_1_latest"] == pytest.approx(0.75, abs=0.0001)
    assert results["values"]["f1_score_1_latest"] == pytest.approx(0.66666, abs=0.0001)
