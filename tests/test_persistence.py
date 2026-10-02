"""Saving and loading must reproduce a network exactly."""
import os
import tempfile
import numpy as np

from nn import (Dense, Tanh, Sigmoid, ReLU, Standardizer, Adam, binary_cross_entropy,
                binary_cross_entropy_prime, predict, train, save_model, load_model)


def _trained_network():
    np.random.seed(0)
    x, y = np.random.randn(80, 4), np.random.randint(0, 2, size=(80, 1)).astype(float)
    net = [Dense(4, 6, init="he", l2=0.01), ReLU(), Dense(6, 3), Tanh(), Dense(3, 1), Sigmoid()]
    train(net, binary_cross_entropy, binary_cross_entropy_prime, x, y, Adam(0.01),
          epochs=20, verbose=False)
    return net, x


def test_round_trip_gives_identical_predictions():
    net, x = _trained_network()
    scaler = Standardizer().fit(x)
    with tempfile.TemporaryDirectory() as d:
        path = save_model(os.path.join(d, "sub", "m.npz"), net, scaler, ["a", "b", "c", "d"], {"acc": 0.7})
        np.random.seed(999)                                   # prove nothing depends on RNG state
        m = load_model(path)
    assert np.array_equal(predict(net, x), predict(m.network, x))
    assert np.array_equal(scaler.transform(x), m.scaler.transform(x))
    assert m.feature_names == ["a", "b", "c", "d"] and m.metadata == {"acc": 0.7}
    assert [type(l).__name__ for l in m.network] == [type(l).__name__ for l in net]
    assert m.network[0].l2 == 0.01


def test_loaded_model_can_keep_training():
    net, x = _trained_network()
    y = np.random.randint(0, 2, size=(80, 1)).astype(float)
    with tempfile.TemporaryDirectory() as d:
        m = load_model(save_model(os.path.join(d, "m"), net))     # ".npz" added automatically
    before = [p.copy() for l in m.network for p in l.params()]
    train(m.network, binary_cross_entropy, binary_cross_entropy_prime, x, y, Adam(0.01),
          epochs=5, verbose=False)
    after = [p for l in m.network for p in l.params()]
    assert any(not np.array_equal(b, a) for b, a in zip(before, after))


def test_model_without_scaler_loads():
    net, _ = _trained_network()
    with tempfile.TemporaryDirectory() as d:
        m = load_model(save_model(os.path.join(d, "m.npz"), net))
    assert m.scaler is None and m.feature_names is None


def test_unsupported_layer_is_rejected():
    from nn import Activation
    try:
        save_model("unused.npz", [Activation(np.sin, np.cos)])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_missing_file_gives_clear_error():
    try:
        load_model("definitely_not_here.npz")
    except FileNotFoundError as e:
        assert "No saved model" in str(e)
        return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
