"""This module provides functions for encoding target labels and metrcics-toolbox-
specific utilities.

The toolbox expects 2D arrays for all classification tasks, including binary
classification. However, sklearn's label_binarize returns a 1D array for binary
classification. This module includes a fix to ensure consistent 2D output shape.
"""

import numpy as np
from sklearn.preprocessing import label_binarize


def toolbox_binarize_labels(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Binarize target labels.

    Uses sklearn's label_binarize under the hood,
    with a fix for binary case. Sklearn's label_binarize returns (N, 1)
    shape for binary classification, while the toolbox expects always 2D arrays,
    to align metrics compute methods to work consistently for binary, multi-class and regression tasks.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels.
    classes : array-like of shape (n_classes,)
        List of all classes from model.classes_.
    Returns
    -------
    Y : ndarray of shape (n_samples, n_classes)
        Binarized target labels.
    """
    y = np.asarray(y)
    Y = label_binarize(y, classes=classes)

    # Fix binary case: convert (N,1) → (N,2)
    # Sklearn uses [false, true] convention for binary case
    # and this has to match with predicted probabilities
    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])

    return Y


def toolbox_binarize_probs(y_pred: np.ndarray) -> np.ndarray:
    """Binarize predicted probabilities for binary classification.

    Ensures that predicted probabilities are in 2D array format.
    If input is 1D, converts it to 2D with two columns: [false, true],
    aligning with sklearn's convention. Already 2D inputs are returned unchanged.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.

    Returns
    -------
    y_pred_binarized : ndarray of shape (n_samples, n_classes)
        Binarized predicted probabilities.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = np.column_stack([1 - y_pred, y_pred])  # [false, true] convention
    return y_pred


def toolbox_widen_series(y: np.ndarray) -> np.ndarray:
    """Widen regression targets to 2D array format.

    Converts 1D regression target arrays to 2D format with a single column,
    ensuring consistency across all metric computations in the toolbox.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Regression target values.

    Returns
    -------
    y_widened : ndarray of shape (n_samples, n_targets)
        Widened regression target values.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Convert to (n_samples, 1)
    return y
