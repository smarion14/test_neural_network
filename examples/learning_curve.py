"""Does more data help?  Train on growing slices of the training games, score on the same test games.

If a model's curve has flattened, more games of the same kind would not make it better; the limit is in
the features. The validation set keeps its full size at every point (it is needed to stop training early),
so the small slices get a better early-stopping signal than they would in real life.

Run from the project root:   python -m examples.learning_curve        (about 5 minutes)
Prints a table and saves docs/learning_curve.png.
"""
import numpy as np
import matplotlib.pyplot as plt

from nn import (Adam, Dense, GradientBoosting, LogisticRegression, ReLU, Sigmoid, Standardizer, accuracy,
                binary_cross_entropy, binary_cross_entropy_prime, load_lol, predict, train, train_val_test_split)
from nn.visualize import plot_learning_curve

CSV_PATH = "data/high_diamond_ranked_10min.csv"
FRACTIONS = [0.03, 0.06, 0.12, 0.25, 0.5, 1.0]           # of the training games
SPLITS = 5

X, y, _ = load_lol(CSV_PATH)
scores = {"Logistic regression": [], "Neural network": [], "Boosted trees": []}      # name -> splits x sizes

for s in range(SPLITS):
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = train_val_test_split(X, y, seed=s)
    scaler = Standardizer().fit(x_tr)                     # one scaling for every slice
    s_va, s_te = scaler.transform(x_va), scaler.transform(x_te)
    sizes = [int(len(x_tr) * f) for f in FRACTIONS]
    for name in scores:
        scores[name].append([])
    for m in sizes:                                       # the split is already shuffled: a prefix is a random sample
        a, b = x_tr[:m], y_tr[:m]
        s_a = scaler.transform(a)
        scores["Logistic regression"][-1].append(
            accuracy(y_te, LogisticRegression().fit(s_a, b).predict(s_te)))
        np.random.seed(s)                                 # same seed -> reproducible weight init
        net = [Dense(X.shape[1], 16, init="he", l2=1e-4), ReLU(), Dense(16, 8, init="he", l2=1e-4), ReLU(),
               Dense(8, 1), Sigmoid()]
        train(net, binary_cross_entropy, binary_cross_entropy_prime, s_a, b, Adam(learning_rate=0.001),
              x_val=s_va, y_val=y_va, epochs=300, batch_size=64, patience=25, verbose=False, seed=s)
        scores["Neural network"][-1].append(accuracy(y_te, predict(net, s_te)))
        scores["Boosted trees"][-1].append(accuracy(y_te, GradientBoosting().fit(a, b, x_va, y_va).predict(x_te)))
    print(f"split {s + 1}/{SPLITS} done", flush=True)

print(f"\nTest accuracy, mean of {SPLITS} splits ({len(x_te):,} test games each)\n")
print("| Training games | " + " | ".join(scores) + " |")
print("|---|" + "---|" * len(scores))
for j, m in enumerate(sizes):
    print(f"| {m:,} | " + " | ".join(f"{np.mean([run[j] for run in scores[name]]):.1%}" for name in scores) + " |")

plot_learning_curve(sizes, scores).savefig("docs/learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()
