"""
This module contains the:

- **ReducerEnum:** The main end user enumeration of available metric reducer types.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum

from .reducers import (
    LatestReducer,
    MaxReducer,
    MeanReducer,
    MinMaxReducer,
    MinReducer,
    StdReducer,
)


class ReducerEnum(Enum):
    LATEST = LatestReducer()
    MEAN = MeanReducer()
    STD = StdReducer()
    MAX = MaxReducer()
    MIN = MinReducer()
    MINMAX = MinMaxReducer()
