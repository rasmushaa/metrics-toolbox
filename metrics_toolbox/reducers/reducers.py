from abc import ABC, abstractmethod

import numpy as np

from .enums import ReducerEnum


# --------------------------- Base MetricReducer --------------------------- #
class MetricReducer(ABC):
    """A common interface for metric reducers.

    Reducers take a list of float values and reduce them to a single float value, by
    calling the __call__ method.
    """

    _name: ReducerEnum

    def __repr__(self):
        return f"MetricReducer(name={self._name})"

    @property
    def name(self) -> ReducerEnum:
        return self._name

    @abstractmethod
    def __call__(self, values: list[float]) -> float:
        pass


# --------------------------- Implementations --------------------------- #
class LatestReducer(MetricReducer):
    _name = ReducerEnum.LATEST

    def __call__(self, values: list[float]) -> float:
        return values[-1]


class MeanReducer(MetricReducer):
    _name = ReducerEnum.MEAN

    def __call__(self, values: list[float]) -> float:
        return float(np.mean(values))


class StdReducer(MetricReducer):
    _name = ReducerEnum.STD

    def __call__(self, values: list[float]) -> float:
        return float(np.std(values))


class MaxReducer(MetricReducer):
    _name = ReducerEnum.MAX

    def __call__(self, values: list[float]) -> float:
        return float(np.max(values))


class MinReducer(MetricReducer):
    _name = ReducerEnum.MIN

    def __call__(self, values: list[float]) -> float:
        return float(np.min(values))


class MinMaxReducer(MetricReducer):
    _name = ReducerEnum.MINMAX

    def __call__(self, values: list[float]) -> float:
        return float(np.max(values) - np.min(values))
