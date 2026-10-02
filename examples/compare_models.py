"""Does the hidden layer help?  (logistic regression vs the 13-16-8-1 network)

Trains each model on several different random train/validation/test splits of the
same data and compares test scores. Using several splits matters: with ~1,500 test
games, accuracy on any single split has a standard error of about 1.2 points
(a 95% margin of about +/- 2.3).

Run from the project root:   python -m examples.compare_models
Takes under a minute. Prints a table you can paste into the README and saves comparison.png.
"""
import numpy as np
import matplotlib.pyplot as plt

from nn import (Dense, ReLU, Sigmoid, Adam, binary_cross_entropy, binary_cross_entropy_prime,
                predict, train, accuracy, majority_baseline, load_lol,
                train_val_test_split, Standardizer)
from nn.visualize import plot_model_comparison

CSV_PATH = "data/high_diamond_ranked_10min.csv"
N_SPLITS = 5


def logistic_regression(n_features):
    """No hidden layer: one Dense unit + sigmoid. Each feature gets one weight."""
    return [Dense(n_features, 1), Sigmoid()], Adam(learning_rate=0.01)


def small_network(n_features):
    """The same network as examples/lol_winner.py."""
    net = [Dense(n_features, 16, init="he", l2=1e-4), ReLU(),
           Dense(16, 8, init="he", l2=1e-4), ReLU(),
           Dense(8, 1), Sigmoid()]
    return net, Adam(learning_rate=0.001)


MODELS = {"Logistic regression\n(no hidden layer)": logistic_regression,
          "Network\n13-16-8-1": small_network}

X, y, names = load_lol(CSV_PATH)
acc = {m: [] for m in MODELS}
logloss = {m: [] for m in MODELS}
n_params = {}
baseline = []

for s in range(N_SPLITS):
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = train_val_test_split(X, y, seed=s)
    scaler = Standardizer().fit(x_tr)
    x_tr, x_va, x_te = (scaler.transform(a) for a in (x_tr, x_va, x_te))
    baseline.append(majority_baseline(y_te))

    line = f"split {s + 1}/{N_SPLITS}:  majority {baseline[-1]:.3f}"
    for model_name, build in MODELS.items():
        np.random.seed(s)                                   # same seed -> reproducible init
        net, optimizer = build(X.shape[1])
        train(net, binary_cross_entropy, binary_cross_entropy_prime, x_tr, y_tr, optimizer,
              x_val=x_va, y_val=y_va, epochs=300, batch_size=64, patience=25,
              verbose=False, seed=s)
        pred = predict(net, x_te)
        acc[model_name].append(accuracy(y_te, pred))
        logloss[model_name].append(binary_cross_entropy(y_te, pred))
        n_params[model_name] = sum(p.size for layer in net for p in layer.params())
        line += f"  |  {model_name.splitlines()[0]} {acc[model_name][-1]:.3f}"
    print(line)

# ---- summary ----
def ms(v):
    v = np.asarray(v)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"

print(f"\nResults over {N_SPLITS} random splits (mean ± standard deviation of the test set score)\n")
print("| Model | Parameters | Test accuracy | Test log loss |")
print("|---|---|---|---|")
print(f"| Always guess majority | 0 | {ms(baseline)} | n/a |")
for m in MODELS:
    print(f"| {m.replace(chr(10), ' ')} | {n_params[m]} | {ms(acc[m])} | {ms(logloss[m])} |")

first, second = list(MODELS)
d_acc = np.array(acc[second]) - np.array(acc[first])
d_ll = np.array(logloss[second]) - np.array(logloss[first])
print(f"\nNetwork minus logistic regression, per split (paired):")
print(f"  accuracy : {d_acc.mean():+.4f} ± {d_acc.std(ddof=1):.4f}   "
      f"(network ahead on {int((d_acc > 0).sum())} of {N_SPLITS} splits)")
print(f"  log loss : {d_ll.mean():+.4f} ± {d_ll.std(ddof=1):.4f}   (negative = network better)")
print("\nNote: both models use early stopping on validation loss; their hyperparameters were not tuned separately.")

plot_model_comparison(acc, baseline, title="Test accuracy on 5 random splits").savefig(
    "comparison.png", dpi=150, bbox_inches="tight")
plt.show()
