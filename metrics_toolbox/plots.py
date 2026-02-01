"""This module contains plotting utilities for visualizing metrics and evaluation
results."""

from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .metrics.results import MetricResult


def plot_confusion_matrix(accuracy_results: List[MetricResult]) -> plt.Figure:
    """Plot confusion matrices from accuracy metrics metadata.

    Parameters
    ----------
    accuracy_results : List[MetricResult]
        A list of MetricResult objects containing confusion matrix data in their metadata.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the plotted confusion matrices. Closed to prevent display upon creation.
    """
    MAX_COLUMNS = 3
    n_matrices = len(accuracy_results)
    n_cols = max(
        1, min(n_matrices, MAX_COLUMNS)
    )  # From 1 to <max> columns, depending on number of matrices
    n_rows = (n_matrices + n_cols - 1) // n_cols  # Rows as needed to fit all matrices

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=120)

    # Flatten ax array for easy indexing, even if there's only one subplot
    if n_matrices == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    for i, result in enumerate(accuracy_results):
        cm = result.metadata["confusion_matrix"]
        classes = result.metadata.get("class_names", np.arange(len(cm)))
        normalize = result.metadata.get("confusion_normalization", None)

        # Set colorbar limits if normalized
        im_kw = {}
        if normalize is not None:
            im_kw = {"vmin": 0.0, "vmax": 1.0}

        # Use ConfusionMatrixDisplay for sklearn-like appearance
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(
            ax=ax[i], cmap="Greens", colorbar=True, im_kw=im_kw, values_format=".2f"
        )
        disp.ax_.set_title(f"Confusion Matrix (Normalize on {normalize})")

    # Hide extra subplots if any
    for j in range(n_matrices, len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_auc_curves(
    auc_metrics: Dict[str, List[MetricResult]], is_roc: bool = True
) -> plt.Figure:
    """Plot ROC or PR curves for given AUC metrics.

    Parameters
    ----------
    auc_metrics : Dict[str, List[MetricResult]]
        A dictionary mapping metric names to lists of MetricResult objects containing
        the AUC values and corresponding FPR/TPR or Recall/Precision data.
    is_roc : bool, optional
        If True, labels the plot for ROC curves; if False, for PR curves. Default is True.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the plotted curves. Closed to prevent display upon creation.
    """
    MAX_COLUMNS = 3
    n_metrics = len(auc_metrics)
    n_cols = max(
        1, min(n_metrics, MAX_COLUMNS)
    )  # From 1 to <max> columns, depending on number of metrics
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Rows as needed to fit all metrics

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=120)

    # Flatten ax array for easy indexing, even if there's only one subplot
    if n_metrics == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    for i, (metric_name, results_list) in enumerate(auc_metrics.items()):

        values = [r.value for r in results_list]
        fprs = [r.metadata["fpr"] for r in results_list]
        tprs = [r.metadata["tpr"] for r in results_list]

        # Plot each ROC curve fold
        for fold, (fpr, tpr) in enumerate(zip(fprs, tprs)):
            label = (
                rf"$\overline{{AUC}} = {np.mean(values):.2f}$" if fold == 0 else None
            )
            ax[i].plot(
                fpr,
                tpr,
                label=label,
                color="darkgreen",
                ls="--",
                alpha=max(
                    0.4, (1.0 - (len(values) - 1) * 0.2)
                ),  # More folds -> less alpha
            )

        # Figure settings
        ax[i].plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.5)
        ax[i].set_xlabel("False Positive Rate" if is_roc else "Recall")
        ax[i].set_ylabel("True Positive Rate" if is_roc else "Precision")
        ax[i].set_title(metric_name)
        ax[i].grid(ls="--", alpha=0.7, color="gray")
        ax[i].legend()

    # Hide extra subplots if any
    for j in range(n_metrics, len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    plt.close(fig)
    return fig
