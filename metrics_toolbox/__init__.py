"""This is the main package for the metrics toolbox.

It automatically imports all necessary modules and classes for easy access:

- MetricEnum: Enumeration of available metrics.
- ReducerEnum: Enumeration of available reducers.
- EvaluatorBuilder: Class to build evaluators using specified metrics and reducers.

Users can directly import from this package to utilize the metrics toolbox functionalities.
"""

from .builder import EvaluatorBuilder
from .metrics.enums import MetricEnum
from .reducers.enums import ReducerEnum
