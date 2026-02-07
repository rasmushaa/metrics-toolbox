import inspect
from typing import List, Optional, Sequence

from .evaluator import MetricEvaluator
from .metrics.registry import MetricEnum
from .reducers.registry import ReducerEnum
from .spec import MetricSpec
from .utils import value_to_enum


class EvaluatorBuilder:
    """A builder for constructing MetricEvaluator instances.

    This builder allows for step-by-step configuration of the MetricEvaluator, including
    adding metrics with specific reducers and options. Or, alternatively, loading
    configuration from a dictionary.
    """

    def __init__(self) -> None:
        self._metric_specs: List[MetricSpec] = []

    def __repr__(self) -> str:
        val = "EvaluatorBuilder(\nmetric_specs=["
        for spec in self._metric_specs:
            val += f"\n\t{spec}"
        val += "\n])"
        return val

    def add_metric(
        self,
        metric: str | MetricEnum,
        reducers: Sequence[ReducerEnum] = (ReducerEnum.LATEST,),
        **kwargs,
    ) -> "EvaluatorBuilder":
        """Add a Metric to the evaluator.

        Metric can be specified by name or enum.

        Details
        -------
        By default, the metric is added with the LATEST reducer and no options.
        Reducers and options can be specified via kwargs.
        Reducers can be provided as strings or ReducerEnum values.

        Parameters
        ----------
        metric : str|MetricEnum
            The Metric name or enum to add.
        reducers : Sequence[ReducerEnum], optional
            The reducers to apply to the metric results, by default (ReducerEnum.LATEST,)
        **kwargs : dict, optional
            Supported options are metrics dependent, and raise TypeError if unsupported options
            are provided. See individual Metric documentation or the error message for details.

        Returns
        -------
        EvaluatorBuilder
            The builder instance for chaining.
        """
        reducers = tuple(
            value_to_enum(r, ReducerEnum) for r in reducers  # Reducer enums are classes
        )
        metric_cls = value_to_enum(metric, MetricEnum).value  # Metric enums are classes

        # Validate kwargs against metric class __init__ parameters
        sig = inspect.signature(metric_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        invalid_params = set(kwargs.keys()) - valid_params

        if invalid_params:
            raise TypeError(
                f"Metric '{metric_cls.__name__}' got unexpected keyword argument(s): {', '.join(sorted(invalid_params))}. "
                f"Valid parameters are: {', '.join(sorted(valid_params))}"
            )

        self._metric_specs.append(
            MetricSpec(metric_cls_instantiated=metric_cls(**kwargs), reducers=reducers)
        )
        return self

    def from_dict(self, cfg: dict) -> "EvaluatorBuilder":
        """Configure the builder from a dictionary.

        Parameters
        ----------
        cfg : dict
            The configuration dictionary.

        Returns
        -------
        EvaluatorBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> cfg = {
        ...     "<metric name>": {"reducers": ["mean", "min"], <metric specific kwargs>: ...},
        ... }
        >>> builder = EvaluatorBuilder().from_dict(cfg)
        """
        for metric_name, kwargs in cfg.items():
            self.add_metric(metric_name, **kwargs)
        return self

    def build(
        self, class_to_instantiate: Optional[type[MetricEvaluator]] = None
    ) -> MetricEvaluator:
        """Execute the builder and produce a MetricEvaluator.

        Parameters
        ----------
        class_to_instantiate : Optional[type[MetricEvaluator]], optional
            The MetricEvaluator class to instantiate, by default MetricEvaluator

        Returns
        -------
        MetricEvaluator
            The constructed MetricEvaluator instance with the specified metrics.
        """
        if class_to_instantiate is None:
            class_to_instantiate = MetricEvaluator
        return class_to_instantiate(metric_specs=self._metric_specs)
