from .enums import MetricNameEnum, MetricScopeEnum
from .results import MetricResult


class Metric:
    """The base class for all metrics.

    Each Metric subclass must implement the compute method,
    and define the class attributes:
    - _name: The name of the metric.
    - _requires_probs: Whether the metric uses probability predictions (floats).
    - _requires_labels: Whether the metric uses label predictions (integers/strings).
    - _requires_classes: Whether the metric requires class information (multi-class metrics).
    - _scope: The scope of the metric (binary, micro, macro, class for logging).
    """

    _name: MetricNameEnum = None
    _scope: MetricScopeEnum = None
    _requires_probs: bool = False
    _requires_labels: bool = False
    _requires_classes: bool = False

    def __repr__(self) -> str:
        return f"Metric(name={self.name}, scope={self.scope}, requires_probs={self.requires_probs}, requires_labels={self.requires_labels}, requires_classes={self.requires_classes})"

    @property
    def name(self) -> MetricNameEnum:
        return self._name

    @property
    def requires_probs(self) -> bool:
        return self._requires_probs

    @property
    def requires_labels(self) -> bool:
        return self._requires_labels

    @property
    def requires_classes(self) -> bool:
        return self._requires_classes

    @property
    def scope(self) -> MetricScopeEnum:
        return self._scope

    def compute(self, y_true, y_pred, classes=None, class_name=None) -> MetricResult:
        raise NotImplementedError
