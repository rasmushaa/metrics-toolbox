"""Reducer registry for available metric reducers.

This module contains the REDUCER_REGISTRY which maps reducer type enums to their
corresponding reducer implementations.

To add new reducers, update the REDUCER_REGISTRY with the new reducer type and class
name.
"""

from .enums import ReducerEnum
from .reducers import (
    LatestReducer,
    MaxReducer,
    MeanReducer,
    MinMaxReducer,
    MinReducer,
    StdReducer,
)

REDUCER_REGISTRY = {
    ReducerEnum.LATEST: LatestReducer(),
    ReducerEnum.MEAN: MeanReducer(),
    ReducerEnum.STD: StdReducer(),
    ReducerEnum.MAX: MaxReducer(),
    ReducerEnum.MIN: MinReducer(),
    ReducerEnum.MINMAX: MinMaxReducer(),
}
