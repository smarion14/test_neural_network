import numpy as np

from .dense import Dense
from .activations import ReLU, Sigmoid
from .losses import binary_cross_entropy, binary_cross_entropy_prime
from .network import predict, train
from .optimizers import Adam
from .data import Standardizer
from .boosting import GradientBoosting
from .symmetry import augment, mirror


class Ensemble:
    """Logistic regression + neural network + gradient-boosted trees, averaged.

    The three models make different kinds of mistakes (linear vs smooth non-linear vs
    axis-aligned splits), so their average usually scores better than any one of them.
    shortcut: plain unweighted mean, fit weights (or a stacker) on validation if one model dominates.

    signs (see nn.symmetry): when given, every model trains on both views of each game and
    answers with the average of its view and the mirror's, so it cannot depend on which team is blue.
    """

    def __init__(self, signs=None):
        self.signs = signs

    def fit(self, x_tr, y_tr, x_va, y_va, seed=0):
        if self.signs is not None:
            (x_tr, y_tr), (x_va, y_va) = augment(x_tr, y_tr, self.signs), augment(x_va, y_va, self.signs)
        self.scaler = Standardizer().fit(x_tr)
        s_tr, s_va = self.scaler.transform(x_tr), self.scaler.transform(x_va)
        n = x_tr.shape[1]

        np.random.seed(seed)                      # same seed -> reproducible weight init
        self.logreg = [Dense(n, 1), Sigmoid()]
        self._train(self.logreg, s_tr, y_tr, s_va, y_va, lr=0.01, seed=seed)
        self.net = [Dense(n, 16, init="he", l2=1e-4), ReLU(),
                    Dense(16, 8, init="he", l2=1e-4), ReLU(),
                    Dense(8, 1), Sigmoid()]
        self._train(self.net, s_tr, y_tr, s_va, y_va, lr=0.001, seed=seed)
        self.trees = GradientBoosting().fit(x_tr, y_tr, x_va, y_va)
        return self

    @staticmethod
    def _train(network, x_tr, y_tr, x_va, y_va, lr, seed):
        train(network, binary_cross_entropy, binary_cross_entropy_prime, x_tr, y_tr,
              Adam(learning_rate=lr), x_val=x_va, y_val=y_va, epochs=300, batch_size=64,
              patience=25, verbose=False, seed=seed)

    def _models(self, X):
        s = self.scaler.transform(X)
        return {"Logistic regression": predict(self.logreg, s),
                "Neural network": predict(self.net, s),
                "Boosted trees": self.trees.predict(X)}

    def predict_all(self, X):
        """Probabilities from each model and from their average."""
        out = self._models(X)
        if self.signs is not None:
            other = self._models(mirror(X, self.signs))
            out = {name: 0.5 * (p + 1 - other[name]) for name, p in out.items()}
        out["Ensemble"] = np.mean(list(out.values()), axis=0)
        return out
