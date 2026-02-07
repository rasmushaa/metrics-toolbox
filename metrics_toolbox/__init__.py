"""This is the main package for the metrics toolbox.

It automatically imports all necessary modules and classes for easy access:

- **MetricEnum:** Enumeration of available metrics.

- **ReducerEnum:** Enumeration of available reducers.

- **EvaluatorBuilder:** Class to build evaluators using specified metrics and reducers.

Users can directly import from this package to utilize the metrics toolbox functionalities.

The API reference and readme can be found at: https://rasmushaa.github.io/metrics-toolbox/
or using the __docs_url__ variable.

Example:
    ```python
    import metrics_toolbox
    from metrics_toolbox import MetricEnum, ReducerEnum, EvaluatorBuilder
    url = metrics_toolbox.__docs_url__
    ```
"""

__docs_url__ = "https://rasmushaa.github.io/metrics-toolbox/"

from .builder import EvaluatorBuilder
from .metrics.registry import MetricEnum
from .reducers.registry import ReducerEnum
