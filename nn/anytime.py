import numpy as np

from .boosting import GradientBoosting
from .data import Standardizer
from .logistic import LogisticRegression
from .symmetry import augment, mirror


class Anytime:
    """One logistic regression and one set of boosted trees that answer at any minute of the game.

    They train on rows from every minute of every training game (see timeline.build_rows), with the
    minute as the last column. The logistic regression sees each feature, each feature times the minute
    and the minute itself, so a lead can count for more late in the game than early; the trees find
    that kind of interaction on their own.

    signs (see nn.symmetry), one per feature, makes both models train on every game from both teams'
    points of view and answer with the average of the two. The minute column is the same for both teams.
    """

    def __init__(self, signs=None):
        self.signs = None if signs is None else np.append(signs, 1.0)

    def fit(self, X, y, X_val, y_val):
        if self.signs is not None:
            (X, y), (X_val, y_val) = augment(X, y, self.signs), augment(X_val, y_val, self.signs)
        self.scaler = Standardizer().fit(X[:, :-1])
        self.logreg = LogisticRegression(ridge=1.0).fit(self._design(X), y)
        self.trees = GradientBoosting().fit(X, y, X_val, y_val)
        return self

    def _design(self, X):
        z = self.scaler.transform(X[:, :-1])
        t = ((X[:, -1] - 15) / 10)[:, None]                       # minute 15 -> 0, one unit = ten minutes
        return np.hstack([z, z * t, t])

    def _models(self, X):
        return {"Logistic regression": self.logreg.predict(self._design(X)),
                "Boosted trees": self.trees.predict(X)}

    def predict_all(self, X):
        """Blue's win probability from each model and from their average, for rows built by build_rows."""
        out = self._models(X)
        if self.signs is not None:
            other = self._models(mirror(X, self.signs))
            out = {name: 0.5 * (p + 1 - other[name]) for name, p in out.items()}
        out["Both"] = np.mean(list(out.values()), axis=0)
        return out
