from dataclasses import dataclass
from typing import Any, Dict, Optional

from .enums import MetricNameEnum, MetricScopeEnum, MetricTypeEnum


@dataclass(frozen=True)
class MetricResult:
    """The result of a metric computation.

    Each MetricResult contains:

    - name: The name of the metric.

    - scope: The scope of the metric (binary, micro, macro, class).

    - type: The type of the metric (probs, labels, etc).

    - value: The computed metric value.

    - metadata: Optional additional information about the computation.

    - options: Optional dictionary of options used during computation.
    """

    name: MetricNameEnum
    scope: MetricScopeEnum
    type: MetricTypeEnum
    value: float
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
