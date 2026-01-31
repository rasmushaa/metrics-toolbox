from abc import ABC, abstractmethod

import numpy as np


# --------------------------- Base MetricReducer --------------------------- #
class MetricReducer(ABC):
    """A common interface for metric reducers.

    Reducers take a list of float values and reduce them to a single float value, by
    calling the apply method.
    """

    @abstractmethod
    def apply(self, values: list[float]) -> float:
        pass


# --------------------------- Implementations --------------------------- #
class LatestReducer(MetricReducer):
    """Reducer that returns the latest value from the list."""

    def apply(self, values: list[float]) -> float:
        return values[-1]


class MeanReducer(MetricReducer):
    """Reducer that returns the mean of the values."""

    def apply(self, values: list[float]) -> float:
        return float(np.mean(values))


class StdReducer(MetricReducer):
    """Reducer that returns the standard deviation of the values."""

    def apply(self, values: list[float]) -> float:
        return float(np.std(values))


class MaxReducer(MetricReducer):
    """Reducer that returns the maximum value from the list."""

    def apply(self, values: list[float]) -> float:
        return float(np.max(values))


class MinReducer(MetricReducer):
    """Reducer that returns the minimum value from the list."""

    def apply(self, values: list[float]) -> float:
        return float(np.min(values))


class MinMaxReducer(MetricReducer):
    """Reducer that returns the difference between the maximum and minimum values."""

    def apply(self, values: list[float]) -> float:
        return float(np.max(values) - np.min(values))
