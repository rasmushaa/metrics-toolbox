from .enums import MetricNameEnum, MetricScopeEnum
from .results import MetricResult


class Metric:
    """The base class for all metrics.

    Each Metric subclass must implement the compute method,
    and define the class attributes to specify what kind of
    metric it is.

    Attributes
    ----------
    _name: MetricNameEnum
        The name of the metric.
    _scope: MetricScopeEnum
        The scope of the metric (binary, micro, macro, class for logging).
    _requires_probs: bool
        Whether the metric uses probability predictions (floats).
    _requires_labels: bool
        Whether the metric uses label predictions (integers/strings).
    _requires_classes: bool
        Whether the metric requires class information (multi-class metrics).
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
        """The name of the metric.

        Returns
        -------
        MetricNameEnum
            The name of the metric.
        """
        return self._name

    @property
    def scope(self) -> MetricScopeEnum:
        """The scope of the metric.

        Returns
        -------
        MetricScopeEnum
            The scope of the metric.
        """
        return self._scope

    @property
    def id(self) -> str:
        """Unique identifier for the metric instance.

        Combines the metric name and scope to create a unique ID,
        matching the MetricEnum naming convention.

        Returns
        -------
        str
            The unique identifier for the metric instance.
        """
        return f"{self.name.value}_{self.scope.value}"

    @property
    def requires_probs(self) -> bool:
        """Whether the metric requires probability predictions.

        Intended to be used in the evaluator to determine
        if probability predictions need to be computed.

        Returns
        -------
        bool
            True if the metric requires probability predictions.
        """
        return self._requires_probs

    @property
    def requires_labels(self) -> bool:
        """Whether the metric requires label predictions.

        Intended to be used in the evaluator to determine
        if label predictions need to be computed.

        Returns
        -------
        bool
            True if the metric requires label predictions.
        """
        return self._requires_labels

    @property
    def requires_classes(self) -> bool:
        """Whether the metric requires class information.

        Intended to be used in the evaluator to determine
        if class information needs to be provided by the user.

        Returns
        -------
        bool
            True if the metric requires class information.
        """
        return self._requires_classes

    def compute(self, y_true, y_pred, classes=None, class_name=None) -> MetricResult:
        """A method to compute the metric."""
        raise NotImplementedError
