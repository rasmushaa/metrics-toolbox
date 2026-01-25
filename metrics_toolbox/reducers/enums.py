"""
This module contains the:
- MetricReducerEnum: An enumeration of available metric reducer types.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum


class MetricReducerEnum(Enum):
    LATEST = "latest"
    MEAN = "mean"
    STD = "std"
    MAX = "max"
    MIN = "min"
    MINMAX = "minmax"
