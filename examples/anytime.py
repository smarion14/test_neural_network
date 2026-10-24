"""One model for every minute of the game, against one model per checkpoint minute.

Pooling the rows of every minute gives a model many more rows to learn from (the same game at minutes
5, 6, 7 ... 30), and the pooled model can answer at ANY minute, not only at the checkpoints. The folds
are split by game, so the rows of one game never land in both the training and the test side.
The pooled models train on both teams' views of each game (nn/symmetry.py).

Run from the project root:   python -m examples.anytime        (needs data/games.jsonl from collect.riot)
"""
import numpy as np

from nn import (Anytime, ChampionPrior, GradientBoosting, LogisticRegression, MIRROR_SIGNS, Standardizer,
                accuracy, binary_cross_entropy, build_rows, load_games)

GAMES_PATH = "data/games.jsonl"
FOLDS, BOOTSTRAPS = 5, 2000
MINUTES = range(5, 31)                           # the minutes the pooled models train on
CHECKPOINTS = (5, 10, 15, 20, 25)                # the minutes every model is scored at

games = load_games(GAMES_PATH)
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(len(games)), FOLDS)
answers = {}                                     # (model, checkpoint) -> [(probabilities, labels) for each fold]


def keep(model, minute, p, y):
    answers.setdefault((model, minute), []).append((p, y))


for k, test in enumerate(folds):
    rest = rng.permutation(np.concatenate([f for j, f in enumerate(folds) if j != k]))
    n_val = int(0.15 * len(rest))
    train_games, val_games, test_games = ([games[i] for i in ids] for ids in (rest[n_val:], rest[:n_val], test))
    prior = ChampionPrior().fit(train_games)                    # from the training games only

    X, y, _, _ = build_rows(train_games, MINUTES, prior, training=True)
    X_val, y_val, _, _ = build_rows(val_games, MINUTES, prior)
    pooled = Anytime(MIRROR_SIGNS).fit(X, y, X_val, y_val)      # mirrored: see nn/symmetry.py

    for minute in CHECKPOINTS:
        X_te, y_te, _, _ = build_rows(test_games, [minute], prior)
        if len(X_te) == 0:
            continue
        for kind, p in pooled.predict_all(X_te).items():
            keep(f"{kind}, all minutes", minute, p, y_te)
        X_m, y_m, _, _ = build_rows(train_games, [minute], prior, training=True)
        X_mv, y_mv, _, _ = build_rows(val_games, [minute], prior)
        X_m, X_mv, X_one = X_m[:, :-1], X_mv[:, :-1], X_te[:, :-1]      # a model per minute has no use for the minute
        scaler = Standardizer().fit(X_m)
        lr = LogisticRegression(ridge=1.0).fit(scaler.transform(X_m), y_m).predict(scaler.transform(X_one))
        trees = GradientBoosting().fit(X_m, y_m, X_mv, y_mv).predict(X_one)
        for kind, p in (("Logistic regression", lr), ("Boosted trees", trees), ("Both", (lr + trees) / 2)):
            keep(f"{kind}, one per minute", minute, p, y_te)
    print(f"fold {k + 1}/{FOLDS} done", flush=True)

stacked = {key: (np.vstack([p for p, _ in runs]), np.vstack([t for _, t in runs])) for key, runs in answers.items()}
models = sorted({m for m, _ in stacked}, key=lambda m: ("one per minute" not in m, m))
n_games = {minute: len(stacked[(models[0], minute)][1]) for minute in CHECKPOINTS}

print(f"\nTest accuracy / log loss at each minute, {FOLDS}-fold cross-validation split by game\n")
print("| Model | " + " | ".join(f"minute {m} ({n_games[m]} games)" for m in CHECKPOINTS) + " |")
print("|---|" + "---|" * len(CHECKPOINTS))
for model in models:
    cells = []
    for minute in CHECKPOINTS:
        p, t = stacked[(model, minute)]
        cells.append(f"{accuracy(t, p):.1%} / {binary_cross_entropy(t, p):.3f}")
    print(f"| {model} | " + " | ".join(cells) + " |")

print("\nAccuracy gain of the all-minutes model over a model per minute (points; 95% interval over games)\n")
print("| Model | " + " | ".join(f"minute {m}" for m in CHECKPOINTS) + " |")
print("|---|" + "---|" * len(CHECKPOINTS))
for kind in ("Logistic regression", "Boosted trees", "Both"):
    cells = []
    for minute in CHECKPOINTS:
        (p, t), (q, _) = stacked[(f"{kind}, all minutes", minute)], stacked[(f"{kind}, one per minute", minute)]
        d = ((p >= 0.5) == (t >= 0.5)).astype(float).ravel() - ((q >= 0.5) == (t >= 0.5)).astype(float).ravel()
        lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAPS)], [2.5, 97.5])
        cells.append(f"{d.mean() * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]")
    print(f"| {kind} | " + " | ".join(cells) + " |")
