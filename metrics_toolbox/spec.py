import inspect
from typing import Dict, List, Sequence

import numpy as np

from metrics_toolbox.metrics.results import MetricResult

from .metrics.base_metric import Metric
from .reducers.registry import ReducerEnum


class MetricSpec:
    """A specification for a metric to be computed in the evaluator.

    Contains the instantiated Metric class,
    the reducers to apply to the metric results,
    and the history of computed MetricResults.

    Parameters
    ----------
    metric_cls_instantiated : Metric
        The instantiated Metric to compute.
    reducers : Sequence[ReducerEnum], optional
        The reducers to apply to the metric results.
        Default is (ReducerEnum.LATEST,).
    """

    def __init__(
        self,
        metric_cls_instantiated: Metric,
        reducers: Sequence[ReducerEnum] = (ReducerEnum.LATEST,),
    ):
        self.__metric_cls = metric_cls_instantiated
        self.__reducers = reducers
        self.__history: List[MetricResult] = []

    def __repr__(self) -> str:
        reducer_names = ", ".join([r.name.lower() for r in self.__reducers])
        return f"MetricSpec(cls={self.__metric_cls}, reducers=({reducer_names}))"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """Compute the metric using the specified options.

        Calls the compute method of the Metric class
        with all provided **kwargs options.
        Appends the result to the Spec history.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        **kwargs : Any
            Additional options to pass to the Metric's compute method.
            Supported options depend on the specific Metric. See the Metric documentation for details.
            Extra options are allowed even if not used by the Metric.
            Metrics themselves will not support unknown options, and will raise errors if unsupported options are passed.
        """
        # Get the signature of the metric's compute method
        compute_sig = inspect.signature(self.__metric_cls.compute)

        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in compute_sig.parameters
        }

        self.__history.append(
            self.__metric_cls.compute(y_true=y_true, y_pred=y_pred, **filtered_kwargs)
        )

    def get_reduced_values(self) -> Dict[str, float]:
        """Get the reduced values for the metric using the specified reducers.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping reducer names to their reduced values.
        """
        reduced_values = {}
        for reducer_enum in self.__reducers:
            name = f"{self.id}_{reducer_enum.name.lower()}"
            value = reducer_enum.value.apply(self.get_values_history())
            reduced_values[name] = value
        return reduced_values

    def get_results_history(self) -> List[MetricResult]:
        """Get the history of MetricResults computed by this spec.

        Returns
        -------
        List[MetricResult]
            The list of MetricResults.
        """
        return self.__history

    def get_values_history(self) -> List[float]:
        """Get the history of metric values computed by this spec.

        Returns
        -------
        List[float]
            The list of metric values.
        """
        return [r.value for r in self.__history]

    def clear_history(self) -> None:
        """Clear the history of MetricResults."""
        self.__history = []

    @property
    def reducers(self) -> Sequence[ReducerEnum]:
        """The reducers applied to this metric spec."""
        return self.__reducers

    @property
    def metric(self) -> Metric:
        """The Metric instance of this metric spec."""
        return self.__metric_cls

    @property
    def id(self) -> str:
        """A unique identifier for the metric spec.

        Note, the same Metrics with different reducers will have the same id.

        Returns
        -------
        str
            The unique identifier.
        """
        return self.__metric_cls.id
