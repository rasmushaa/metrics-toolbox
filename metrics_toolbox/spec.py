from dataclasses import dataclass, field
from typing import Optional, Sequence

from .metrics.base import Metric
from .reducers.enums import MetricReducerEnum


@dataclass(frozen=True)
class MetricSpec:
    """A specification for a metric to be computed in the evaluator.

    Contains the Metric class and any additional parameters
    to compute the metric with.

    Parameters
    ----------
    metric_cls : Type[Metric]
        The Metric class to compute.
    reducers : Sequence[MetricReducerEnum], optional
        The reducers to apply to the metric results, by default (MetricReducerEnum.LATEST,)
    class_name : Optional[str], optional
        The class name for class-specific metrics, by default None.
    """

    metric_cls: Metric
    reducers: Sequence[MetricReducerEnum] = field(
        default_factory=lambda: (MetricReducerEnum.LATEST,)
    )
    class_name: Optional[str] = None

    def __repr__(self) -> str:
        reducers_str = ", ".join([r.value for r in self.reducers])
        return f"MetricSpec(metric_cls={self.metric_cls}, reducers=({reducers_str}), class_name={self.class_name})"

    @property
    def id(self) -> str:
        """A unique identifier for the metric spec.

        The id is baesed on the metric class name and optional class name.
        Note, the same Metrics with different reducers will have the same id.

        Returns
        -------
        str
            The unique identifier.
        """
        base = self.metric_cls.name.value
        if self.class_name is not None:
            return f"{base}_{self.class_name}"
        return base
