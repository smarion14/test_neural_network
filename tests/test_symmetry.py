import numpy as np

from nn import Ensemble, augment, mirror, symmetric

SIGNS = np.array([-1.0, -1.0, -1.0, 1.0])            # three blue-minus-red features and one the same for both teams


def test_mirror_flips_the_differences_and_keeps_the_shared_feature():
    X = np.array([[2.0, -1.0, 3.0, 7.0]])
    assert np.array_equal(mirror(X, SIGNS), [[-2.0, 1.0, -3.0, 7.0]])
    assert np.array_equal(mirror(mirror(X, SIGNS), SIGNS), X)               # swapping twice changes nothing


def test_augment_adds_each_game_seen_from_the_other_side():
    X, y = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), np.array([[1.0], [0.0]])
    Xa, ya = augment(X, y, SIGNS)
    assert Xa.shape == (4, 4) and np.array_equal(Xa[:2], X)
    assert np.array_equal(Xa[2:], mirror(X, SIGNS)) and ya.ravel().tolist() == [1, 0, 0, 1]   # blue won -> red won


def test_symmetric_answers_add_up_to_one_for_any_predictor():
    rng = np.random.default_rng(0)
    w, b = rng.normal(size=4), 0.7                                           # biased on purpose: not symmetric by itself
    biased = lambda X: 1 / (1 + np.exp(-(X @ w + b))).reshape(-1, 1)
    X = rng.normal(size=(50, 4))
    assert not np.allclose(biased(X) + biased(mirror(X, SIGNS)), 1)
    assert np.allclose(symmetric(biased, X, SIGNS) + symmetric(biased, mirror(X, SIGNS), SIGNS), 1)


def test_mirrored_ensemble_cannot_tell_which_team_is_blue():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=200) > 0).astype(float).reshape(-1, 1)
    ens = Ensemble(signs=SIGNS).fit(X[:140], y[:140], X[140:], y[140:], seed=0)
    here, there = ens.predict_all(X[140:]), ens.predict_all(mirror(X[140:], SIGNS))
    for name in here:
        assert np.allclose(here[name] + there[name], 1), name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
