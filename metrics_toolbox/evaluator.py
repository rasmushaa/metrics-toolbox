from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from .metrics.base import MetricResult
from .reducers.registry import REDUCER_REGISTRY


class MetricEvaluator:
    """Evaluates and tracks metrics over multiple updates.

    You can create a MetricEvaluator with a list of MetricSpecs, or use the
    EvaluatorBuilder for a more convenient interface.
    """

    def __init__(self, metric_specs):
        """Initialize the MetricEvaluator with a list of MetricSpecs.

        Duplicate MetricSpecs (same metric and class_name) are not allowed,
        and will raise a ValueError.

        Parameters
        ----------
        metric_specs : list[MetricSpec]
            A list of MetricSpec instances defining the metrics to evaluate.
        """
        self._metric_specs = metric_specs
        self._history: Dict[str, list[MetricResult]] = {}
        self.__validate_metric_specs()

    def __repr__(self) -> str:
        val = "MetricEvaluator(\nmetric_specs=[\n"
        for spec in self._metric_specs:
            val += "\t" + repr(spec) + ",\n"
        val += "])"
        return val

    def update_model_evaluation(self, model, X, y_true):
        """Update the evaluation based on model predictions.

        Parameters
        ----------
        model : Any
            A model with predict and optionally predict_proba methods.
        X : np.ndarray
            Input features for prediction.
        y_true : np.ndarray
            True labels. Ints or strings.
        """
        y_pred = model.predict(X)
        self.update_labels_evaluation(y_true, y_pred, classes=model.classes)
        y_pred_probs = model.predict_proba(X)
        self.update_probs_evaluation(y_true, y_pred_probs, classes=model.classes)

    def update_labels_evaluation(
        self, y_true: np.ndarray, y_pred: np.ndarray[np.integer | np.str_], classes=None
    ):
        """Update the evaluation for metrics that require labels.

        Parameters
        ----------
        y_true : np.ndarray
            True labels. Ints or strings.
        y_pred : np.ndarray[np.integer | np.str_]
            Predicted labels. Ints or strings.
        classes : list, optional
            List of class labels, required for some metrics.
        """
        for spec in self._metric_specs:
            if spec.metric_cls.requires_labels:
                args = {}
                if spec.metric_cls.requires_classes:
                    args["classes"] = classes
                if spec.class_name:
                    args["class_name"] = spec.class_name
                result = spec.metric_cls.compute(y_true, y_pred, **args)
                self._history.setdefault(spec.id, []).append(result)

    def update_probs_evaluation(
        self, y_true: np.ndarray, y_pred: np.ndarray[np.float64], classes=None
    ):
        """Update the evaluation for metrics that require probabilities.

        Parameters
        ----------
        y_true : np.ndarray
            True labels. Ints or strings.
        y_pred : np.ndarray[np.float64]
            Predicted probabilities in floats.
        classes : list, optional
            List of class labels, required for some metrics.
        """
        for spec in self._metric_specs:
            if spec.metric_cls.requires_probs:
                args = {}
                if spec.metric_cls.requires_classes:
                    args["classes"] = classes
                if spec.class_name:
                    args["class_name"] = spec.class_name
                result = spec.metric_cls.compute(y_true, y_pred, **args)
                self._history.setdefault(spec.id, []).append(result)

    def results(self) -> Dict[str, Dict[str, float | list[float] | plt.Figure]]:
        """Get the evaluation results, including reduced metrics and full history.

        Returns
        -------
        Dict[str, Dict[str, float | list[float] | plt.Figure]]
            A dictionary with 'reduced' and 'history' keys containing metric results,
            where reduced metrics are aggregated using specified reducers.
            Also includes matplotlib Figures for metrics that produce plots.
        """
        summary: Dict[str, Dict[str, float | list[float] | plt.Figure]] = {
            "reduced": {},
            "history": {},
            "figures": {},
        }
        for spec in self._metric_specs:

            # Metric values over time
            values = [r.value for r in self._history.get(spec.id, [])]

            base_name = spec.metric_cls.name.value
            if spec.class_name:
                base_name += f"_{spec.class_name}"

            # 1. Add the reduced metric to summary. LATEST is the default (not reduced) and does not need suffix.
            for reducer in spec.reducers:
                summary_key = f"{base_name}_{reducer.value}"
                summary["reduced"][summary_key] = REDUCER_REGISTRY[reducer](values)

            # 2. Add the full history as well
            summary["history"][f"{base_name}_steps"] = values
        return summary

    def __validate_metric_specs(self):
        """Chekc that the metric specs do not contain duplicate entries."""
        seen_ids = set()
        for spec in self._metric_specs:
            if spec.id in seen_ids:
                raise ValueError(f"Duplicate MetricSpec id found: {spec.id}")
            seen_ids.add(spec.id)
