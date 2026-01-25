from enum import Enum
from typing import Type, List

from .spec import MetricSpec
from .metrics.base import Metric
from .reducers.registry import MetricReducerEnum
from .metrics.registry import METRIC_REGISTRY, MetricNameEnum
from .evaluator import MetricEvaluator
from .utils import value_to_enum


class EvaluatorBuilder:
    """A builder for constructing MetricEvaluator instances.

    This builder allows for step-by-step configuration of the MetricEvaluator, including
    adding metrics with specific reducers and class names.

    Or, alternatively, loading configuration from a dictionary.
    """

    def __init__(self):
        self._metric_specs: List[MetricSpec] = []

    def __repr__(self) -> str:
        val = "EvaluatorBuilder(\nmetric_specs=["
        for spec in self._metric_specs:
            val += f"\n\t{spec}"
        val += "\n])"
        return val

    def add_metric(self, metric: str | MetricNameEnum, **kwargs):
        """Add a Metric to the evaluator.

        Metric can be specified by name or enum.

        Details
        -------
        By default, the metric is added with the LATEST reducer and no class_name.
        Reducers and class_name can be specified via kwargs.
        Reducers can be provided as strings or MetricReducerEnum values.

        Parameters
        ----------
        metric : str|MetricNameEnum
            The Metric name or enum to add.
        **kwargs : dict, optional
            Additional parameters for the MetricSpec, by default {}
            Supported parameters are listed in MetricSpec.

        Returns
        -------
        EvaluatorBuilder
            The builder instance for chaining.
        """
        if kwargs.get("reducers") is not None:
            kwargs["reducers"] = tuple(
                value_to_enum(r, MetricReducerEnum) for r in kwargs["reducers"]
            )
        metric_name = value_to_enum(metric, MetricNameEnum)
        self._metric_specs.append(
            MetricSpec(metric_cls=METRIC_REGISTRY[metric_name.value](), **kwargs)
        )
        return self

    def from_dict(self, cfg: dict):
        """Configure the builder from a dictionary.

        Parameters
        ----------
        cfg : dict
            The configuration dictionary.
            Must contain a "metrics" key with a list of metric specifications.

        Example
        -------
        {
            "metrics": [
                {
                    "name": "MetricClassName",
                    "reducers": ["mean", "min"],    # Optional
                    "class_name": "A"               # Optional
                },
            ]
        }

        Returns
        -------
        EvaluatorBuilder
            The builder instance for chaining.
        """
        for args in cfg["metrics"]:
            metric_name = args["name"]
            kwargs = {k: v for k, v in args.items() if k != "name"}
            self.add_metric(metric_name, **kwargs)
        return self

    def build(self):
        """Execute the builder and produce a MetricEvaluator.

        Returns
        -------
        MetricEvaluator
            The constructed MetricEvaluator instance with the specified metrics.
        """
        return MetricEvaluator(metric_specs=self._metric_specs)
