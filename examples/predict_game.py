"""Predict a game's winner with the saved model (no retraining).

First run  python -m examples.lol_winner  once to create models/lol_model.npz. Then:

    python -m examples.predict_game goldDiff=1500 expDiff=800 killDiff=3 firstBlood=1

Every stat is BLUE minus RED (negative = red is ahead). firstBlood is 1 if blue got
first blood, 0 if red did. Stats you leave out default to 0.
"""
import argparse
import numpy as np

from nn import load_model, predict

parser = argparse.ArgumentParser(description="Predict a League game from 10-minute stats.")
parser.add_argument("stats", nargs="*", help="feature=value pairs, e.g. goldDiff=1500")
parser.add_argument("--model", default="models/lol_model.npz", help="path to the saved model")
args = parser.parse_args()

try:
    model = load_model(args.model)
except FileNotFoundError:
    raise SystemExit(f"No saved model at '{args.model}'. Run  python -m examples.lol_winner  first.")

values = dict.fromkeys(model.feature_names, 0.0)
for item in args.stats:
    name, sep, raw = item.partition("=")
    if not sep or name not in values:
        raise SystemExit(f"Can't read '{item}'. Use name=value with a name from:\n  "
                         + ", ".join(model.feature_names))
    try:
        values[name] = float(raw)
    except ValueError:
        raise SystemExit(f"'{raw}' is not a number (in '{item}').")

row = np.array([[values[n] for n in model.feature_names]])
p_blue = float(predict(model.network, model.scaler.transform(row))[0, 0])

print("Game stats (blue minus red):")
for n in model.feature_names:
    print(f"  {n:<16}{values[n]:>9.1f}")
favored, p = ("Blue", p_blue) if p_blue >= 0.5 else ("Red", 1 - p_blue)
print(f"\nBlue win chance: {p_blue:.1%}   ->  {favored} favored ({p:.1%})")
if model.metadata:
    print(f"(model test accuracy on held-out games: {model.metadata.get('test_accuracy', float('nan')):.1%})")
