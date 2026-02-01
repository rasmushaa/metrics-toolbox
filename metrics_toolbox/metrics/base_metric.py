import numpy as np

from .enums import MetricNameEnum, MetricScopeEnum, MetricTypeEnum
from .results import MetricResult


class Metric:
    """The base class for all metrics.

    Details
    -------

    - Each Metric subclass must implement the compute method.

    - Define the class private attributes to specify what kind of metric it is

    - Allow to pass possible options during initialization.

    Attributes
    ----------
    _name: MetricNameEnum
        The name of the metric.
    _scope: MetricScopeEnum
        The scope of the metric (micro, macro, class).
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
    def type(self) -> MetricTypeEnum:
        """The type of data the metric operates on.

        Returns
        -------
        MetricTypeEnum
            The type of data the metric operates on.
        """
        return self._type

    @property
    def id(self) -> str:
        """Unique identifier for the metric instance.

        Child classes can override this to make more specific ids if needed.

        Returns
        -------
        str
            The unique identifier for the metric instance.
        """
        return f"{self.name.value}_{self.scope.value}"

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, column_names: list[str]
    ) -> MetricResult:
        """A method to compute the metric.

        Different metrics will have different implementations of this method,
        but they all share the same signature.

        y_true and y_pred are always 2D arrays with the same shape,
        where each column corresponds to a different prediction value,
        and column_names specify the names of these columns.

        It is possible to use Metrics directly, but the assumed way is to use them
        via a Evaluator class, and thus, all input validation is done there.

        Parameters
        ----------
        y_true : np.ndarray
            The ground truth target values.
        y_pred : np.ndarray
            The predicted target values.
        column_names : list[str]
            The names of the columns the input arrays correspond to.
        """
        raise NotImplementedError
