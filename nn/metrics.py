import numpy as np


def accuracy(y_true, y_pred, threshold=0.5):
    """Fraction of predictions on the correct side of the threshold."""
    return float(np.mean((y_pred >= threshold) == (y_true >= 0.5)))


def confusion_matrix(y_true, y_pred, threshold=0.5):
    """[[true 0, false 1], [false 0, true 1]]  (rows = actual, columns = predicted)."""
    t = (y_true >= 0.5).ravel()
    p = (y_pred >= threshold).ravel()
    return np.array([[np.sum(~t & ~p), np.sum(~t & p)],
                     [np.sum(t & ~p), np.sum(t & p)]])


def confidence_table(y_true, y_pred, thresholds=(0.6, 0.7, 0.8, 0.9)):
    """How far to trust a call: for each threshold, the share of games the model is at least that sure
    about, and its accuracy on those games. "Sure" is the probability of the side it picks (0.5 = coin flip).

    Returns rows of (threshold, share of games, accuracy). A threshold no game reaches is left out.
    """
    p = np.asarray(y_pred).ravel()
    sure = np.maximum(p, 1 - p)
    right = (p >= 0.5) == (np.asarray(y_true).ravel() >= 0.5)
    return [(t, float(np.mean(sure >= t)), float(right[sure >= t].mean())) for t in thresholds if np.any(sure >= t)]


def majority_baseline(y_true):
    """Accuracy of always guessing the most common class. A model must beat this."""
    p = float(np.mean(y_true >= 0.5))
    return max(p, 1 - p)
