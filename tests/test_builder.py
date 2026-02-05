import pytest

from metrics_toolbox.builder import EvaluatorBuilder
from metrics_toolbox.metrics.base_metric import MetricScopeEnum
from metrics_toolbox.metrics.enums import MetricNameEnum
from metrics_toolbox.metrics.registry import MetricEnum
from metrics_toolbox.reducers.registry import ReducerEnum


def test_evaluator_builder_add_metric_default():
    """Test that add_metric prduces correct default MetricSpec."""
    builder = EvaluatorBuilder().add_metric(MetricEnum.ROC_AUC_TARGET, target_name=1)

    spec = builder._metric_specs[0]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_1"
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"


def test_evaluator_builder_add_metric_with_params():
    """Test that add_metric correctly handles parameters."""
    builder = (
        EvaluatorBuilder()
        .add_metric(
            MetricEnum.ROC_AUC_TARGET,
            reducers=(ReducerEnum.MEAN, ReducerEnum.MIN),
            target_name=1,
        )
        .add_metric(
            MetricEnum.ROC_AUC_TARGET,
            reducers=("MinMax", "lAtEst"),  # Test string input for reducers
            target_name="positive",
        )
    )
    print(builder)

    spec = builder._metric_specs[0]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_1"
    assert spec.reducers == (ReducerEnum.MEAN, ReducerEnum.MIN)

    spec = builder._metric_specs[1]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_positive"
    assert spec.reducers == (
        ReducerEnum.MINMAX,
        ReducerEnum.LATEST,
    ), "String reducers should be converted to Enum"


def test_evaluator_builder_from_dict():
    """Test that from_dict correctly builds MetricSpecs."""
    cfg = {
        "metrics": [
            {
                "name": "ROC_AUC_TARGET",
                # No reducers or class_name
                "target_name": 1,
            },
            {
                "name": "rOC_AUC_mAcRO",
                "reducers": ["MiN", "mInMaX"],
            },
            {"name": "roc_auc_target", "target_name": "B"},
            {"name": "accuRacy"},
            {"name": "precision_macro"},
        ]
    }

    # Create builder from dict, and chain add_metric
    builder = EvaluatorBuilder().from_dict(cfg).add_metric(MetricEnum.ROC_AUC_MICRO)
    print(builder)

    spec = builder._metric_specs[0]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_1"
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"

    spec = builder._metric_specs[1]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.MACRO.value
    assert spec.reducers == (ReducerEnum.MIN, ReducerEnum.MINMAX)

    spec = builder._metric_specs[2]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_B"
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"
    assert spec.metric.target_name == "B"

    assert builder._metric_specs[3].id == MetricNameEnum.ACCURACY.value
    assert (
        builder._metric_specs[4].id
        == MetricNameEnum.PRECISION.value + "_" + MetricScopeEnum.MACRO.value
    )

    spec = builder._metric_specs[5]
    assert spec.id == MetricNameEnum.ROC_AUC.value + "_" + MetricScopeEnum.MICRO.value
    assert spec.reducers == (
        ReducerEnum.LATEST,
    ), "Default spec reducer should be LATEST"


def test_evaluator_builder_add_metric_with_unsuported_params():
    """Test that bad inputs raise appropriate errors."""
    builder = EvaluatorBuilder()
    with pytest.raises(ValueError):
        builder.add_metric(MetricEnum.ROC_AUC_TARGET, reducers=("unsupported_reducer",))

    with pytest.raises(ValueError):
        builder.add_metric(MetricEnum.ROC_AUC_TARGET, reducers=(123,))

    with pytest.raises(ValueError):
        builder.add_metric(MetricEnum.ROC_AUC_TARGET, reducers=(None,))

    # The add_metric takes **kwargs, and those are passed to MetricSpec.
    with pytest.raises(TypeError, match="Valid parameters are"):
        builder.add_metric(MetricEnum.ROC_AUC_TARGET, non_existing_param=True)
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
                {"name": "ROC_AUC_TARGET", "reducers": "should_be_a_list"}  # Wrong type
            ]
        }
        builder.from_dict(conf)
