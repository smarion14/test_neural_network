import numpy as np

# Each loss returns a scalar E (averaged over every element of the batch).
# Each *_prime returns dE/dY_pred with the same shape as y_pred.
# The division by y_true.size means the gradient is already averaged over the
# batch, so the learning rate behaves the same for any batch size.


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size
