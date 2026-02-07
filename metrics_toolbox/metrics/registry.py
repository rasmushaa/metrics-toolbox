"""
This module contains the:

- **MetricEnum:** The main end user enumeration for metrics.

To build a MetricEvaluator from a configuration,
you have to use the Enumerator values as keys.
"""

from enum import Enum

from .classification.accuracy import Accuracy
from .classification.f1_score_macro import F1ScoreMacro
from .classification.f1_score_micro import F1ScoreMicro
from .classification.f1_score_target import F1ScoreTarget
from .classification.precision_macro import PrecisionMacro
from .classification.precision_micro import PrecisionMicro
from .classification.precision_target import PrecisionTarget
from .classification.recall_macro import RecallMacro
from .classification.recall_micro import RecallMicro
from .classification.recall_target import RecallTarget
from .probability.roc_auc_macro import RocAucMacro
from .probability.roc_auc_micro import RocAucMicro
from .probability.roc_auc_target import RocAucTarget
from .regression.mse_macro import MSEMacro
from .regression.mse_target import MSETarget
from .regression.rmse_macro import RMSEMacro
from .regression.rmse_target import RMSETarget


class MetricEnum(Enum):
    """Enumeration of available metrics."""

    ACCURACY = Accuracy

    F1_SCORE_MICRO = F1ScoreMicro
    F1_SCORE_MACRO = F1ScoreMacro
    F1_SCORE_TARGET = F1ScoreTarget

    PRECISION_MICRO = PrecisionMicro
    PRECISION_MACRO = PrecisionMacro
    PRECISION_TARGET = PrecisionTarget

    RECALL_MICRO = RecallMicro
    RECALL_MACRO = RecallMacro
    RECALL_TARGET = RecallTarget

    ROC_AUC_MICRO = RocAucMicro
    ROC_AUC_MACRO = RocAucMacro
    ROC_AUC_TARGET = RocAucTarget

    MSE_TARGET = MSETarget
    MSE_MACRO = MSEMacro
    RMSE_TARGET = RMSETarget
    RMSE_MACRO = RMSEMacro
