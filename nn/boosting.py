import numpy as np


class GradientBoosting:
    """Binary gradient-boosted trees (Newton boosting on log loss), NumPy only.

    Features are cut into `n_bins` quantile bins once, so finding the best split of a
    node is a histogram of gradients per bin instead of a sort. Each tree is a complete
    binary tree of fixed `depth` stored as arrays (heap order: children of i are
    2i+1 and 2i+2). A node that cannot split keeps its value and acts as a leaf.
    """

    def __init__(self, n_rounds=300, depth=3, lr=0.1, n_bins=32, l2=1.0, min_leaf=20):
        self.n_rounds, self.depth, self.lr = n_rounds, depth, lr
        self.n_bins, self.l2, self.min_leaf = n_bins, l2, min_leaf

    def _bin(self, X):
        return np.column_stack([np.searchsorted(e, X[:, j], side="left")
                                for j, e in enumerate(self.edges)]).astype(np.int32)

    def _grow(self, B, g, h):
        size = 2 ** (self.depth + 1) - 1
        feat, thr, val = np.full(size, -1), np.zeros(size, int), np.zeros(size)

        def build(node, rows, level):
            G, H = g[rows].sum(), h[rows].sum()
            val[node] = -self.lr * G / (H + self.l2)
            if level == self.depth or len(rows) < 2 * self.min_leaf:
                return
            best_gain, best = 1e-9, None
            for j in range(B.shape[1]):
                col = B[rows, j]
                GL = np.cumsum(np.bincount(col, g[rows], self.n_bins))[:-1]
                HL = np.cumsum(np.bincount(col, h[rows], self.n_bins))[:-1]
                nL = np.cumsum(np.bincount(col, minlength=self.n_bins))[:-1]
                gain = (GL ** 2 / (HL + self.l2) + (G - GL) ** 2 / (H - HL + self.l2)
                        - G ** 2 / (H + self.l2))
                gain[(nL < self.min_leaf) | (len(rows) - nL < self.min_leaf)] = 0
                b = int(np.argmax(gain))
                if gain[b] > best_gain:
                    best_gain, best = gain[b], (j, b)
            if best is None:
                return
            j, b = best
            feat[node], thr[node] = j, b
            go_left = B[rows, j] <= b
            build(2 * node + 1, rows[go_left], level + 1)
            build(2 * node + 2, rows[~go_left], level + 1)

        build(0, np.arange(len(B)), 0)
        return feat, thr, val

    def _tree_output(self, tree, B):
        feat, thr, val = tree
        node = np.zeros(len(B), int)
        for _ in range(self.depth):
            f = feat[node]
            right = B[np.arange(len(B)), np.maximum(f, 0)] > thr[node]
            node = np.where(f >= 0, 2 * node + 1 + right, node)
        return val[node]

    @staticmethod
    def _sigmoid(F):
        return 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))

    def fit(self, X, y, x_val=None, y_val=None, patience=20):
        """Boost `n_rounds` trees; with validation data, keep the best round count."""
        y = np.asarray(y, float).ravel()
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        B = self._bin(X)
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base = np.log(p0 / (1 - p0))
        self.trees = []
        F = np.full(len(y), self.base)
        use_val = x_val is not None
        if use_val:
            Bv, yv = self._bin(x_val), np.asarray(y_val, float).ravel()
            Fv = np.full(len(yv), self.base)
        best_loss, best_n = np.inf, 0
        for i in range(self.n_rounds):
            p = self._sigmoid(F)
            tree = self._grow(B, p - y, p * (1 - p))
            self.trees.append(tree)
            F += self._tree_output(tree, B)
            if use_val:
                Fv += self._tree_output(tree, Bv)
                pv = np.clip(self._sigmoid(Fv), 1e-9, 1 - 1e-9)
                loss = -np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
                if loss < best_loss - 1e-9:
                    best_loss, best_n = loss, i + 1
                elif i + 1 - best_n >= patience:
                    break
        if use_val:
            self.trees = self.trees[:best_n]
        return self

    def predict(self, X):
        """Probability that the label is 1, shape (n, 1) like the networks' output."""
        B = self._bin(X)
        F = np.full(len(B), self.base)
        for tree in self.trees:
            F += self._tree_output(tree, B)
        return self._sigmoid(F).reshape(-1, 1)
