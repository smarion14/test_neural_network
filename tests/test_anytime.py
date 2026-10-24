import numpy as np

from nn import Anytime, mirror


def test_one_model_learns_that_the_same_lead_counts_for_more_later():
    rng = np.random.default_rng(0)
    n = 6000
    lead, minute = rng.normal(size=n), rng.integers(5, 31, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-lead * minute / 10))).astype(float).reshape(-1, 1)   # a lead of 1 is worth 0.6 at minute 6
    X = np.column_stack([lead, minute])
    model = Anytime().fit(X[:5000], y[:5000], X[5000:], y[5000:])
    early, late = model.predict_all(np.array([[1.0, 6.0], [1.0, 24.0]]))["Logistic regression"].ravel()
    assert abs(early - 1 / (1 + np.exp(-0.6))) < 0.04 and abs(late - 1 / (1 + np.exp(-2.4))) < 0.04


def test_mirrored_anytime_cannot_tell_which_team_is_blue():
    rng = np.random.default_rng(1)
    n = 1500
    X = np.column_stack([rng.normal(size=(n, 2)), rng.integers(5, 31, size=n)])         # two features, then the minute
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(float).reshape(-1, 1)
    signs = np.array([-1.0, -1.0])
    model = Anytime(signs).fit(X[:1000], y[:1000], X[1000:1200], y[1000:1200])
    here = model.predict_all(X[1200:])
    there = model.predict_all(mirror(X[1200:], model.signs))                           # same games, teams swapped
    for name in here:
        assert np.allclose(here[name] + there[name], 1), name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
