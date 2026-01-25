from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from .enums import MetricNameEnum, MetricScopeEnum


@dataclass(frozen=True)
class MetricResult:
    """The result of a metric computation.

    Each MetricResult contains:
    - name: The name of the metric.
    - scope: The scope of the metric (binary, micro, macro, class).
    - value: The computed metric value.
    - metadata: Optional additional information about the computation.
    - class_name: Optional class name for class-specific metrics.
    """

    name: MetricNameEnum
    scope: MetricScopeEnum
    value: float
    metadata: Optional[Dict[str, Any]] = None
    class_name: Optional[str] = None
