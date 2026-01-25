"""
This module contains the:
- REDUCER_REGISTRY: A registry mapping metric reducer type strings to their corresponding functions.

To add new reducers, update the REDUCER_REGISTRY with the new reducer type and class name.
"""

from .enums import MetricReducerEnum
from .reducers import (
    LatestReducer,
    MeanReducer,
    StdReducer,
    MaxReducer,
    MinReducer,
    MinMaxReducer,
)


REDUCER_REGISTRY = {
    MetricReducerEnum.LATEST: LatestReducer(),
    MetricReducerEnum.MEAN: MeanReducer(),
    MetricReducerEnum.STD: StdReducer(),
    MetricReducerEnum.MAX: MaxReducer(),
    MetricReducerEnum.MIN: MinReducer(),
    MetricReducerEnum.MINMAX: MinMaxReducer(),
}
