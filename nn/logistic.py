import numpy as np


class LogisticRegression:
    """Logistic regression solved exactly with Newton's method: no learning rate, epochs or early stopping.

    Fit it on standardized features. A tiny ridge keeps the Hessian invertible. This is the
    baseline the network and the boosted trees have to beat to be worth their extra parameters.
    shortcut: a fixed number of Newton steps and no convergence check, plenty for a few standardized features.
    """

    def __init__(self, ridge=1e-3, iterations=25):
        self.ridge, self.iterations = ridge, iterations

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])             # first weight is the intercept
        t = np.asarray(y, float).ravel()
        self.w = np.zeros(A.shape[1])
        for _ in range(self.iterations):
            p = self._sigmoid(A @ self.w)
            gradient = A.T @ (p - t) + self.ridge * self.w
            hessian = (A * (p * (1 - p))[:, None]).T @ A + self.ridge * np.eye(len(self.w))
            self.w -= np.linalg.solve(hessian, gradient)
        return self

    def predict(self, X):
        """Probability that the label is 1, shape (n, 1) like the networks' output."""
        return self._sigmoid(np.column_stack([np.ones(len(X)), X]) @ self.w).reshape(-1, 1)
