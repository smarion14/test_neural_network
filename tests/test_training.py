"""Behavioural tests: optimizers, numerical safety, and learning XOR."""
import warnings
import numpy as np

from nn import (Dense, Tanh, Sigmoid, SGD, Adam, binary_cross_entropy,
                binary_cross_entropy_prime, predict, train)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def test_backward_does_not_change_weights():
    """Gradients and updates are separate: backward() alone must leave weights alone."""
    np.random.seed(0)
    layer = Dense(3, 2)
    before = layer.weights.copy()
    layer.forward(np.random.randn(4, 3))
    layer.backward(np.random.randn(4, 2))
    assert np.array_equal(before, layer.weights)


def test_sgd_step_matches_formula():
    p, g = np.array([1.0, 2.0]), np.array([0.5, -1.0])
    SGD(learning_rate=0.1).step([p], [g])
    assert np.allclose(p, [0.95, 2.1])


def test_momentum_accumulates():
    p, g, opt = np.array([0.0]), np.array([1.0]), SGD(learning_rate=0.1, momentum=0.9)
    opt.step([p], [g])          # v = -0.1
    opt.step([p], [g])          # v = 0.9*-0.1 - 0.1 = -0.19
    assert np.allclose(p, [-0.29])


def test_adam_first_step_is_about_lr():
    """With bias correction, Adam's first step has size ~ learning_rate."""
    p = np.array([5.0])
    Adam(learning_rate=0.01).step([p], [np.array([3.0])])
    assert np.allclose(p, [4.99], atol=1e-6)


def test_sigmoid_does_not_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # any overflow warning becomes a failure
        out = Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.allclose(out, [[0.0, 0.5, 1.0]])


def _solve_xor(optimizer, epochs):
    np.random.seed(1)
    net = [Dense(2, 4), Tanh(), Dense(4, 1), Sigmoid()]
    train(net, binary_cross_entropy, binary_cross_entropy_prime, X, Y, optimizer,
          epochs=epochs, verbose=False)
    return (predict(net, X) > 0.5).astype(float)


def test_xor_with_sgd():
    assert np.array_equal(_solve_xor(SGD(0.5), 2000), Y)


def test_xor_with_momentum():
    assert np.array_equal(_solve_xor(SGD(0.2, momentum=0.9), 1000), Y)


def test_xor_with_adam():
    assert np.array_equal(_solve_xor(Adam(0.05), 1000), Y)


def test_early_stopping_restores_best_weights():
    """Train on noise labels so validation loss can only get worse; best epoch must be restored."""
    np.random.seed(5)
    x, y = np.random.randn(60, 4), np.random.randint(0, 2, size=(60, 1)).astype(float)
    xv, yv = np.random.randn(40, 4), np.random.randint(0, 2, size=(40, 1)).astype(float)
    net = [Dense(4, 16), Tanh(), Dense(16, 1), Sigmoid()]
    h = train(net, binary_cross_entropy, binary_cross_entropy_prime, x, y, Adam(0.01),
              x_val=xv, y_val=yv, epochs=300, patience=10, verbose=False)
    assert h["stopped_epoch"] < 300
    assert np.isclose(binary_cross_entropy(yv, predict(net, xv)), min(h["val_loss"]))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
