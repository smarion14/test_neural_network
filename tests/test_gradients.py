"""Numerical gradient checks: compare each layer's analytic backward() to a
finite-difference estimate. If backprop has a bug, these fail.

Run from the project root:   python -m pytest        (or: python -m tests.test_gradients)
"""
import numpy as np

from nn import (Dense, Tanh, Sigmoid, ReLU, mse, mse_prime,
                binary_cross_entropy, binary_cross_entropy_prime, predict)

EPS = 1e-6
TOL = 1e-5


def rel_error(a, b):
    return np.abs(a - b).max() / max(1e-8, np.abs(a).max() + np.abs(b).max())


def numeric_grad(f, array):
    """Central-difference gradient of scalar f() w.r.t. `array`, perturbed in place."""
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        old = array[idx]
        array[idx] = old + EPS
        plus = f()
        array[idx] = old - EPS
        minus = f()
        array[idx] = old
        grad[idx] = (plus - minus) / (2 * EPS)
    return grad


def check_layer(layer, x, penalty=lambda: 0.0):
    """E = sum(R * layer(x)) for a fixed random R, so that dE/dY = R."""
    rng = np.random.default_rng(0)
    R = rng.normal(size=layer.forward(x).shape)
    energy = lambda: np.sum(R * layer.forward(x)) + penalty()

    layer.forward(x)
    dx = layer.backward(R)
    assert rel_error(dx, numeric_grad(energy, x)) < TOL, "input gradient (dE/dX) is wrong"
    for p, g in zip(layer.params(), [g.copy() for g in layer.grads()]):
        assert rel_error(g, numeric_grad(energy, p)) < TOL, "parameter gradient is wrong"


def test_dense_gradients():
    np.random.seed(0)
    check_layer(Dense(4, 3), np.random.randn(5, 4))


def test_dense_gradients_with_l2():
    np.random.seed(1)
    layer = Dense(4, 3, l2=0.3)
    check_layer(layer, np.random.randn(5, 4), penalty=lambda: 0.5 * layer.l2 * np.sum(layer.weights ** 2))


def test_activation_gradients():
    np.random.seed(2)
    for act in (Tanh(), Sigmoid(), ReLU()):
        x = np.random.randn(6, 5)
        x[np.abs(x) < 0.05] += 0.5            # keep ReLU away from its kink at 0
        check_layer(act, x)


def test_loss_gradients():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=(7, 1)).astype(float)
    p = rng.uniform(0.1, 0.9, size=(7, 1))
    for loss, prime in ((mse, mse_prime), (binary_cross_entropy, binary_cross_entropy_prime)):
        assert rel_error(prime(y, p), numeric_grad(lambda: loss(y, p), p)) < TOL


def test_full_network_gradients():
    """End to end: gradients of the real loss w.r.t. every weight in the network."""
    for hidden_act in (Tanh, ReLU):
        np.random.seed(4)
        net = [Dense(3, 5, init="he"), hidden_act(), Dense(5, 4), hidden_act(), Dense(4, 1), Sigmoid()]
        x = np.random.randn(8, 3)
        y = np.random.randint(0, 2, size=(8, 1)).astype(float)

        out = predict(net, x)
        grad = binary_cross_entropy_prime(y, out)
        for layer in reversed(net):
            grad = layer.backward(grad)

        loss = lambda: binary_cross_entropy(y, predict(net, x))
        for layer in net:
            for p, g in zip(layer.params(), [g.copy() for g in layer.grads()]):
                assert rel_error(g, numeric_grad(loss, p)) < TOL


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
