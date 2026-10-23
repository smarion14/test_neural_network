import numpy as np

from nn import LogisticRegression


def test_newton_reaches_the_optimum():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    true_w = np.array([1.5, -2.0, 0.0])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ true_w + 0.5)))).astype(float).reshape(-1, 1)

    model = LogisticRegression().fit(X, y)
    assert np.allclose(model.w, [0.5, *true_w], atol=0.15)                   # intercept comes first
    p = model.predict(X)
    assert p.shape == (4000, 1) and p.min() > 0 and p.max() < 1
    A = np.column_stack([np.ones(len(X)), X])
    gradient = A.T @ (p - y).ravel() + model.ridge * model.w
    assert np.abs(gradient).max() < 1e-6                                     # zero gradient is the optimum


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
