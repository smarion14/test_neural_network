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


def majority_baseline(y_true):
    """Accuracy of always guessing the most common class. A model must beat this."""
    p = float(np.mean(y_true >= 0.5))
    return max(p, 1 - p)
