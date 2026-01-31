import pytest

from metrics_toolbox.builder import EvaluatorBuilder
from metrics_toolbox.metrics.base_metric import Metric, MetricResult, MetricScopeEnum
from metrics_toolbox.metrics.enums import MetricNameEnum
from metrics_toolbox.metrics.registry import MetricEnum
from metrics_toolbox.reducers.registry import ReducerEnum


# ------------------------- Mock Metric -------------------------
class MockMetric(Metric):
    _name = MetricNameEnum.ROC_AUC
    _scope = MetricScopeEnum.BINARY

    def compute(self, y_true, y_pred, **kwargs):
        return MetricResult(
            name=self.name,
            value=0.5,
            metadata={},
            scope=self.scope,
        )


# ------------------------- Tests -------------------------
def test_evaluator_builder_add_metric_default():
    """Test that add_metric prduces correct default MetricSpec."""
    builder = EvaluatorBuilder().add_metric(MetricEnum.ROC_AUC_BINARY)

    spec = builder._metric_specs[0]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.BINARY.value
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"


def test_evaluator_builder_add_metric_with_params():
    """Test that add_metric correctly handles parameters."""
    builder = (
        EvaluatorBuilder()
        .add_metric(
            MetricEnum.ROC_AUC_CLASS,
            reducers=(ReducerEnum.MEAN, ReducerEnum.MIN),
            class_name="A",
        )
        .add_metric(
            MetricEnum.ROC_AUC_BINARY,
            reducers=("MinMax", "lAtEst"),  # Test string input for reducers
        )
    )
    print(builder)

    spec = builder._metric_specs[0]
    assert (
        spec.id
        == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.CLASS.value + "_A"
    )
    assert spec.reducers == (ReducerEnum.MEAN, ReducerEnum.MIN)

    spec = builder._metric_specs[1]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.BINARY.value
    assert spec.reducers == (
        ReducerEnum.MINMAX,
        ReducerEnum.LATEST,
    ), "String reducers should be converted to Enum"


def test_evaluator_builder_from_dict():
    """Test that from_dict correctly builds MetricSpecs."""
    cfg = {
        "metrics": [
            {
                "name": "ROC_AUC_BINARY",
                # No reducers or class_name
            },
            {
                "name": "rOC_AUC_mAcRO",
                "reducers": ["MiN", "mInMaX"],
            },
            {"name": "roc_auc_class", "class_name": "B"},
        ]
    }

    # Create builder from dict, and chain add_metric
    builder = EvaluatorBuilder().from_dict(cfg).add_metric(MetricEnum.ROC_AUC_MICRO)
    print(builder)

    spec = builder._metric_specs[0]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.BINARY.value
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"

    spec = builder._metric_specs[1]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.MACRO.value
    assert spec.reducers == (ReducerEnum.MIN, ReducerEnum.MINMAX)

    spec = builder._metric_specs[2]
    assert (
        spec.id
        == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.CLASS.value + "_B"
    )
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"
    assert spec.metric.class_name == "B"

    spec = builder._metric_specs[3]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.MICRO.value
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"


def test_evaluator_builder_add_metric_with_unsuported_params():
    """Test that bad inputs raise appropriate errors."""
    builder = EvaluatorBuilder()
    with pytest.raises(ValueError):
        builder.add_metric(MetricEnum.ROC_AUC_BINARY, reducers=("unsupported_reducer",))

    with pytest.raises(ValueError):
        builder.add_metric(MetricEnum.ROC_AUC_BINARY, reducers=(123,))

    with pytest.raises(ValueError):
        builder.add_metric(MetricEnum.ROC_AUC_CLASS, reducers=(None,))

    # The add_metric takes **kwargs, and those are passed to MetricSpec.
    with pytest.raises(TypeError):
        builder.add_metric(MetricEnum.ROC_AUC_CLASS, non_existing_param=True)

    with pytest.raises(ValueError):
        conf = {"metrics": [{"name": "non_existing_metric"}]}
        builder.from_dict(conf)

    # Missing name key
    with pytest.raises(KeyError):
        conf = {"metrics": [{"reducers": ["mean"]}]}
        builder.from_dict(conf)

    with pytest.raises(ValueError):
        conf = {
            "metrics": [
                {"name": "ROC_AUC_BINARY", "reducers": "should_be_a_list"}  # Wrong type
            ]
        }
        builder.from_dict(conf)
