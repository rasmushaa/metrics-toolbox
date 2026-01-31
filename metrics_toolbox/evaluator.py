from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

from metrics_toolbox.metrics.enums import MetricNameEnum, MetricTypeEnum
from metrics_toolbox.plots import plot_auc_curves
from metrics_toolbox.spec import MetricSpec


class MetricEvaluator:
    """Evaluates and tracks metrics over multiple updates.

    You can create a MetricEvaluator with a list of MetricSpecs, or use the
    EvaluatorBuilder for a more convenient interface.

    Methods
    -------
    add_prob_evaluation(y_true, y_pred, classes=None)
        Evaluate PROB metrics and add new step to history.
    add_label_evaluation(y_true, y_pred, classes=None)
        Evaluate LABEL metrics and add new step to history.
    add_model_evaluation(model, X, y_true)
        Evaluate a model and all included metrics. This method assumes the model has
        predict() and predict_proba() methods.
    get_results()
        Get evaluation results including reduced values, full history, and plots.
    """

    def __init__(self, metric_specs: List[MetricSpec]):
        """Initialize the MetricEvaluator with a list of MetricSpecs.

        Duplicate MetricSpecs (same metric, scope, and class_name) are not allowed,
        and will raise a ValueError.

        Parameters
        ----------
        metric_specs : list[MetricSpec]
            A list of MetricSpec instances defining the metrics to evaluate.
        """
        self._metric_specs = metric_specs
        self.__validate_metric_specs()

    def __repr__(self) -> str:
        val = "MetricEvaluator(\nmetric_specs=[\n"
        for spec in self._metric_specs:
            val += "  " + repr(spec) + ",\n"
        val += "])"
        return val

    def add_prob_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[List[str]] = None,
    ):
        """Evaluate PROB metrics and add new step to history.

        Parameters
        ----------
        y_true : np.ndarray
            True labels. Ints or strings.
        y_pred : np.ndarray
            Predicted probabilities. Shape (n_samples, n_classes).
        classes : Optional[List[str]], optional
            List of class labels. Needed for some metrics, by default None.
        """
        y_true = np.asarray(y_true)  # Ensure numpy array, if not already
        y_pred = np.asarray(y_pred)

        if not (
            np.issubdtype(y_true.dtype, np.integer)
            or np.issubdtype(y_true.dtype, np.str_)
        ):
            raise ValueError(
                "y_true must contain integers or strings for probabilities"
            )
        if not np.issubdtype(y_pred.dtype, np.floating):
            raise ValueError("y_pred must contain floats for probabilities")
        if not np.all((y_pred >= 0.0) & (y_pred <= 1.0)):
            raise ValueError(
                "y_pred must contain probabilities in the range [0.0, 1.0]"
            )

        prob_specs = self.__get_prob_specs()
        for spec in prob_specs:
            spec.compute(
                y_true=y_true,
                y_pred=y_pred,
                classes=classes,
            )

    def add_label_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[List[str]] = None,
    ):
        """Evaluate LABEL metrics and add new step to history.

        Parameters
        ----------
        y_true : np.ndarray
            True labels. Ints or strings.
        y_pred : np.ndarray
            Predicted labels. Ints or strings.
        classes : Optional[List[str]], optional
            List of class labels. Needed for some metrics, by default None.
        """
        y_true = np.asarray(y_true)  # Ensure numpy array, if not already
        y_pred = np.asarray(y_pred)

        if not (
            np.issubdtype(y_true.dtype, np.integer)
            or np.issubdtype(y_true.dtype, np.str_)
        ):
            raise ValueError("y_true must contain integers or strings for labels")
        if not (
            np.issubdtype(y_pred.dtype, np.integer)
            or np.issubdtype(y_pred.dtype, np.str_)
        ):
            raise ValueError("y_pred must contain integers or strings for labels")

        label_specs = self.__get_label_specs()
        for spec in label_specs:
            spec.compute(
                y_true=y_true,
                y_pred=y_pred,
                classes=classes,
            )

    def add_model_evaluation(self, model, X: np.ndarray, y_true: np.ndarray):
        """Evaluate a model and all included metrics.

        This method does not what your model **predict()** methods returns,
        and it is assumed you only include compatible metrics in the evaluator.
        For example, classifier and regressor have the same predict() method signature,
        but you should not mix classification and regression metrics in the same evaluator.

        Mixing is allowed if you use the lower-level:

        - **add_label_evaluation()**

        - **add_prob_evaluation()**

        and you are able to inherit the Evaluator and crate your own model evaluation logic.
        You can build custom evaluators using the **EvaluatorBuilder**,
        by passing your custom evaluator class to the **build()** method.

        Details
        -------
        This method performs the following steps:
        1. If there are **LABEL metrics** to evaluate,
           it uses model.predict(X) to get predicted labels
           and calls **add_label_evaluation()**.
        2. If there are **PROB metrics** to evaluate,
           it uses model.predict_proba(X) to get predicted probabilities
           and calls **add_prob_evaluation()**.

        Parameters
        ----------
        model : Any
            A model with predict and predict_proba methods.
        X : np.ndarray
            Input features for prediction.
        y_true : np.ndarray
            True labels. Ints or strings.
        """
        classes = self.__get_model_classes(model)

        if self.__get_label_specs():  # If there are LABEL metrics to evaluate
            y_pred = model.predict(X)
            self.add_label_evaluation(
                y_true=y_true,
                y_pred=y_pred,
                classes=classes,
            )

        if self.__get_prob_specs():  # If there are PROB metrics to evaluate
            y_pred = model.predict_proba(X)
            self.add_prob_evaluation(
                y_true=y_true,
                y_pred=y_pred,
                classes=classes,
            )

    def get_results(self) -> Dict[str, Dict[str, float | list[float] | plt.Figure]]:
        """Get evaluation results including reduced values, full history, and plots.

        Returns
        -------
        Dict[str, Dict[str, float | list[float] | plt.Figure]]
            A dictionary with keys 'values', 'steps', and 'figures'.

            - 'values': Reduced metric values (e.g., mean, max).

            - 'steps': Full history of metric values over evaluation steps.

            - 'figures': Plots for applicable metrics (e.g., ROC AUC curves).
        """

        summary: Dict[str, Dict[str, float | list[float] | plt.Figure]] = {
            "values": {},
            "steps": {},
            "figures": {},
        }

        def get_reduced_values(specs) -> Dict[str, float]:
            """Iterate over specs and fill reduced values."""
            reduced = {}
            for spec in specs:
                reduced.update(spec.get_reduced_values())  # {roc_auc_mean: 0.85, ...}
            return reduced

        def get_full_history(specs) -> Dict[str, list[float]]:
            """Iterate over specs ids in history and get full values over given
            specs."""
            history = {}
            for spec in specs:
                history[f"{spec.id}_steps"] = (
                    spec.get_values_history()
                )  # {roc_auc_steps: [0.8, 0.85, ...], ...}
            return history

        def get_roc_auc_plots(specs) -> Dict[str, plt.Figure]:
            """Generate ROC AUC plots for given specs."""
            roc_auc_results = {}
            for spec in specs:
                if spec.metric.name == MetricNameEnum.ROC_AUC:
                    roc_auc_results[spec.id] = spec.get_results_history()
            if roc_auc_results:
                fig = plot_auc_curves(
                    auc_metrics=roc_auc_results,
                    is_roc=True,
                )
                return {"roc_auc_curves": fig}
            return {}

        summary["values"].update(get_reduced_values(self._metric_specs))
        summary["steps"].update(get_full_history(self._metric_specs))
        summary["figures"].update(get_roc_auc_plots(self._metric_specs))
        return summary

    def __validate_metric_specs(self):
        """Check that the metric specs do not contain duplicate entries."""
        seen_ids = set()
        for spec in self._metric_specs:
            if spec.id in seen_ids:
                raise ValueError(f"Duplicate MetricSpec id found: {spec.id}")
            seen_ids.add(spec.id)

    def __get_model_classes(self, model) -> List[str]:
        """Get class labels from the model if available.

        Returns
        -------
        List[str]
            List of class labels.
        """

        if hasattr(model, "classes_"):
            classes = model.classes_
        elif hasattr(model, "classes"):
            classes = model.classes
        else:
            raise ValueError(
                "Model does not have 'classes' or 'classes_' attribute required for some metrics."
            )
        return classes

    def __get_prob_specs(self) -> List[MetricSpec]:
        """Get the list of metric IDs that require probabilities.

        Returns
        -------
        List[str]
            List of metric IDs requiring probabilities.
        """
        return self.__find_specs_by_type(MetricTypeEnum.PROBS)

    def __get_label_specs(self) -> List[MetricSpec]:
        """Get the list of metric IDs that require labels.

        Returns
        -------
        List[str]
            List of metric IDs requiring labels.
        """
        return self.__find_specs_by_type(MetricTypeEnum.LABELS)

    def __find_specs_by_type(self, metric_type: MetricTypeEnum) -> List[MetricSpec]:
        """Find all MetricSpecs of a given type.

        Parameters
        ----------
        metric_type : MetricTypeEnum
            The type of metrics to find.

        Returns
        -------
        List[MetricSpec]
            List of MetricSpecs matching the given type.
        """
        return [spec for spec in self._metric_specs if spec.metric.type == metric_type]
