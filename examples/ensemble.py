"""Does combining models beat the single network?  And how early can we call a game?

    python -m examples.ensemble                              # Kaggle 10-minute CSV
    python -m examples.ensemble --games data/games.jsonl     # your own Riot data, at 5/10/15/20 min

Each model is scored on the same random splits, so the numbers are paired.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from nn import (Ensemble, accuracy, binary_cross_entropy, load_lol, load_games, build_features,
                ChampionPrior, train_val_test_split)

ap = argparse.ArgumentParser()
ap.add_argument("--games", help="games.jsonl from collect.riot (default: the Kaggle CSV)")
ap.add_argument("--splits", type=int, default=3)
args = ap.parse_args()


def evaluate(make_xy):
    """make_xy(split_seed) -> (train, val, test) as (X, y) pairs. Returns {model: [acc per split]}."""
    scores = {}
    for s in range(args.splits):
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = make_xy(s)
        preds = Ensemble().fit(x_tr, y_tr, x_va, y_va, seed=s).predict_all(x_te)
        for name, p in preds.items():
            scores.setdefault(name, []).append((accuracy(y_te, p), binary_cross_entropy(y_te, p)))
    return {m: np.array(v) for m, v in scores.items()}


def show(scores, title):
    print(f"\n{title}\n| Model | Test accuracy | Test log loss |\n|---|---|---|")
    for m, v in scores.items():
        print(f"| {m} | {v[:, 0].mean():.3f} ± {v[:, 0].std(ddof=1):.3f} | {v[:, 1].mean():.3f} |")


if not args.games:
    X, y, _ = load_lol("data/high_diamond_ranked_10min.csv")
    show(evaluate(lambda s: train_val_test_split(X, y, seed=s)), f"Kaggle CSV, {len(X)} games")
else:
    games = load_games(args.games)
    minutes, curves = [5, 10, 15, 20], {}
    for minute in minutes:
        usable = [g for g in games if len(g["frames"]) > minute]      # games that lasted that long

        def make_xy(seed):
            wins = np.array([g["blueWin"] for g in usable], float).reshape(-1, 1)
            parts = train_val_test_split(np.arange(len(usable)).reshape(-1, 1), wins, seed=seed)
            prior = ChampionPrior().fit([usable[i] for i in parts[0][0].ravel()])  # training games only
            return [build_features([usable[i] for i in idx.ravel()], minute, prior, training=(j == 0))[:2]
                    for j, (idx, _) in enumerate(parts)]          # parts[0] is the training set

        scores = evaluate(make_xy)
        show(scores, f"Minute {minute}, {len(usable)} games")
        for m, v in scores.items():
            curves.setdefault(m, []).append(v[:, 0].mean())

    for m, c in curves.items():
        plt.plot(minutes, c, marker="o", label=m, linewidth=3 if m == "Ensemble" else 1.5)
    plt.xlabel("Minutes into the game")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.savefig("docs/accuracy_vs_time.png", dpi=150, bbox_inches="tight")
    plt.show()
