from .metrics.base import MetricResult
from .metrics.enums import MetricScopeEnum
from .reducers.enums import MetricReducerEnum
from .reducers.registry import REDUCER_REGISTRY
from typing import Dict


class MetricEvaluator:
    def __init__(self, metric_specs):
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

        y_pred = model.predict(X)
        self.update_labels_evaluation(y_true, y_pred, classes=model.classes)

        if hasattr(model, "predict_proba"):
            y_pred_probs = model.predict_proba(X)
            self.update_probs_evaluation(y_true, y_pred_probs, classes=model.classes)

    def update_labels_evaluation(self, y_true, y_pred, classes=None):
        for spec in self._metric_specs:
            if spec.metric_cls.requires_labels:
                args = {}
                if spec.metric_cls.requires_classes:
                    args["classes"] = classes
                if spec.class_name:
                    args["class_name"] = spec.class_name
                result = spec.metric_cls.compute(y_true, y_pred, **args)
                self._history.setdefault(spec.id, []).append(result)

    def update_probs_evaluation(self, y_true, y_pred, classes=None):
        for spec in self._metric_specs:
            if spec.metric_cls.requires_probs:
                args = {}
                if spec.metric_cls.requires_classes:
                    args["classes"] = classes
                if spec.class_name:
                    args["class_name"] = spec.class_name
                result = spec.metric_cls.compute(y_true, y_pred, **args)
                self._history.setdefault(spec.id, []).append(result)

    def results(self):
        summary = {"reduced": {}, "history": {}}
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
