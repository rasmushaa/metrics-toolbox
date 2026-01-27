"""Reducer registry for available metric reducers.

This module contains the REDUCER_REGISTRY which maps reducer type enums to their
corresponding reducer implementations.

To add new reducers, update the REDUCER_REGISTRY with the new reducer type and class
name.
"""

from .enums import MetricReducerEnum
from .reducers import (
    LatestReducer,
    MaxReducer,
    MeanReducer,
    MinMaxReducer,
    MinReducer,
    StdReducer,
)

REDUCER_REGISTRY = {
    MetricReducerEnum.LATEST: LatestReducer(),
    MetricReducerEnum.MEAN: MeanReducer(),
    MetricReducerEnum.STD: StdReducer(),
    MetricReducerEnum.MAX: MaxReducer(),
    MetricReducerEnum.MIN: MinReducer(),
    MetricReducerEnum.MINMAX: MinMaxReducer(),
}
