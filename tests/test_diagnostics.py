import numpy as np
from nn import (Dense, Tanh, Sigmoid, Adam, accuracy, binary_cross_entropy,
                binary_cross_entropy_prime, train, overfitting_report)


def _history(**kw):
    h = {"train_loss": [0.6, 0.5, 0.4, 0.3], "val_loss": [0.6, 0.55, 0.56, 0.58],
         "train_metric": [0.65, 0.70, 0.78, 0.85], "val_metric": [0.64, 0.68, 0.67, 0.66],
         "best_epoch": 2, "stopped_epoch": 4}
    h.update(kw)
    return h


def test_report_numbers_at_best_epoch():
    r = overfitting_report(_history(), verbose=False)
    assert r["best_epoch"] == 2
    assert np.isclose(r["metric_gap_points"], 2.0)             # (0.70 - 0.68) * 100
    assert np.isclose(r["loss_gap"], 0.05)                      # 0.55 - 0.50
    assert r["label"] == "small"
    assert r["epochs_after_best"] == 2
    assert np.isclose(r["train_loss_change"], -0.2) and np.isclose(r["val_loss_change"], 0.03)


def test_labels():
    big = _history(train_metric=[0.7, 0.9, 0.95, 0.99], val_metric=[0.6, 0.62, 0.6, 0.6])
    assert overfitting_report(big, verbose=False)["label"] == "large"
    mid = _history(train_metric=[0.7, 0.75, 0.8, 0.9], val_metric=[0.6, 0.69, 0.6, 0.6])
    assert overfitting_report(mid, verbose=False)["label"] == "moderate"


def test_test_score_margin_of_error():
    r = overfitting_report(_history(), test_score=0.7, n_test=1481, verbose=False)
    assert np.isclose(r["test_margin_points"], 1.96 * np.sqrt(0.7 * 0.3 / 1481) * 100)
    assert np.isclose(r["test_minus_val_points"], 2.0)


def test_requires_validation_data():
    try:
        overfitting_report({"train_loss": [0.5, 0.4]}, verbose=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_works_on_a_real_training_history():
    np.random.seed(0)
    x, y = np.random.randn(100, 3), np.random.randint(0, 2, size=(100, 1)).astype(float)
    xv, yv = np.random.randn(50, 3), np.random.randint(0, 2, size=(50, 1)).astype(float)
    net = [Dense(3, 8), Tanh(), Dense(8, 1), Sigmoid()]
    h = train(net, binary_cross_entropy, binary_cross_entropy_prime, x, y, Adam(0.01),
              x_val=xv, y_val=yv, metric=accuracy, epochs=60, patience=10, verbose=False)
    r = overfitting_report(h, test_score=0.5, n_test=50, verbose=False)
    assert "metric_gap_points" in r and "test_margin_points" in r


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
