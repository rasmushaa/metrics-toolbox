from .enums import MetricNameEnum, MetricScopeEnum, MetricTypeEnum
from .results import MetricResult


class Metric:
    """The base class for all metrics.

    Details
    -------

    - Each Metric subclass must implement the compute method

    - Define the class private attributes to specify what kind of metric it is

    - Allow to pass possible options during initialization.

    Attributes
    ----------
    _name: MetricNameEnum
        The name of the metric.
    _scope: MetricScopeEnum
        The scope of the metric (binary, micro, macro, class for logging).
    _type: MetricTypeEnum
        The type of data the metric operates on (probabilities or labels).
    """

    _name: MetricNameEnum = None
    _scope: MetricScopeEnum = None
    _type: MetricTypeEnum = None

    def __init__(self, **kwargs):
        pass

    def __repr__(self) -> str:
        return f"Metric(name={self.name.value}, scope={self.scope.value}, type={self.type.value})"

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
    def type(self) -> MetricTypeEnum:
        """The type of data the metric operates on.

        Returns
        -------
        MetricTypeEnum
            The type of data the metric operates on.
        """
        return self._type

    def compute(self, *args, **kwargs) -> MetricResult:
        """A method to compute the metric.

        Different metrics will have different implementations of this method, and may
        accept different options via *args and **kwargs.
        """
        raise NotImplementedError
