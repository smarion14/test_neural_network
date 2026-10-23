"""One honest score per model, from all 9,879 games instead of a single 1,481-game test set.

10-fold cross-validation: every game is a test game exactly once, and the out-of-fold predictions
are pooled, so the margin of error is about +/-0.9 points instead of +/-2.3. The models differ by
less than a point, so each is also compared with the exact logistic regression on the same games
(a paired bootstrap). "Logistic regression" in the table is the Adam-trained one inside the ensemble.

Run from the project root:   python -m examples.cross_validate        (about 2 minutes)
Prints a table you can paste into the README.
"""
import numpy as np

from nn import Ensemble, LogisticRegression, Standardizer, binary_cross_entropy, load_lol

CSV_PATH = "data/high_diamond_ranked_10min.csv"
FOLDS, BOOTSTRAPS = 10, 2000
BASE = "Exact logistic regression"

X, y, _ = load_lol(CSV_PATH)
n = len(X)
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(n), FOLDS)
oof, train_hit = {}, {}                      # model -> prediction for every game / train accuracy per fold

for k, test in enumerate(folds):
    rest = rng.permutation(np.concatenate([f for j, f in enumerate(folds) if j != k]))
    n_val = int(0.15 * len(rest))
    val, tr = rest[:n_val], rest[n_val:]     # val is only used to stop training early
    scaler = Standardizer().fit(X[tr])
    exact = LogisticRegression().fit(scaler.transform(X[tr]), y[tr])
    ens = Ensemble().fit(X[tr], y[tr], X[val], y[val], seed=k)
    on_test = {BASE: exact.predict(scaler.transform(X[test])), **ens.predict_all(X[test])}
    on_train = {BASE: exact.predict(scaler.transform(X[tr])), **ens.predict_all(X[tr])}
    for name, p in on_test.items():
        oof.setdefault(name, np.zeros((n, 1)))[test] = p
        train_hit.setdefault(name, []).append(np.mean((on_train[name] >= 0.5) == (y[tr] >= 0.5)))
    print(f"fold {k + 1}/{FOLDS} done", flush=True)

hit = {m: ((p >= 0.5) == (y >= 0.5)).ravel().astype(float) for m, p in oof.items()}
print(f"\nCross-validated over {n:,} games, each tested once\n")
print("| Model | Test accuracy | Log loss | Train minus test accuracy | Accuracy minus exact logistic regression (95% interval) |")
print("|---|---|---|---|---|")
for m, p in oof.items():
    acc = hit[m].mean()
    margin = 1.96 * np.sqrt(acc * (1 - acc) / n)
    if m == BASE:
        versus = "-"
    else:
        d = hit[m] - hit[BASE]
        lo, hi = np.percentile([d[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAPS)], [2.5, 97.5])
        versus = f"{d.mean() * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] points"
    print(f"| {m} | {acc:.1%} ± {margin * 100:.1f} | {binary_cross_entropy(y, p):.4f} "
          f"| {(np.mean(train_hit[m]) - acc) * 100:+.1f} points | {versus} |")
