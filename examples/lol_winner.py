"""Predict which team wins a League of Legends game from its first 10 minutes.

1. Download high_diamond_ranked_10min.csv from Kaggle
   ("League of Legends Diamond Ranked Games (10 min)") into data/.
2. Run from the project root:   python -m examples.lol_winner

The trained network is saved to models/lol_model.npz. Set RESUME = True to continue
training that saved model instead of starting from fresh random weights.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from nn import (Dense, ReLU, Sigmoid, binary_cross_entropy, binary_cross_entropy_prime,
                predict, train, Adam, accuracy, confusion_matrix, majority_baseline,
                load_lol, train_val_test_split, Standardizer, save_model, load_model,
                overfitting_report)
from nn.visualize import (plot_history, plot_confusion_matrix, plot_layer_flow,
                          plot_probability_slice)

CSV_PATH = "data/high_diamond_ranked_10min.csv"
MODEL_PATH = "models/lol_model.npz"
RESUME = False          # True: keep training the saved model

np.random.seed(0)

# ---- 1. data: load and split ----
X, y, names = load_lol(CSV_PATH)
(x_tr, y_tr), (x_va, y_va), (x_te, y_te) = train_val_test_split(X, y, seed=0)
print(f"{len(X)} games, {X.shape[1]} features -> train {len(x_tr)}, val {len(x_va)}, test {len(x_te)}")

# ---- 2. model: fresh, or loaded from disk ----
previous_val_loss = None
if RESUME and os.path.exists(MODEL_PATH):
    saved = load_model(MODEL_PATH)
    if saved.feature_names != names:
        raise ValueError("Saved model was trained on different features. Set RESUME = False.")
    network, scaler = saved.network, saved.scaler          # reuse the SAME scaling it learned with
    previous_val_loss = binary_cross_entropy(y_va, predict(network, scaler.transform(x_va)))
    print(f"Resuming from {MODEL_PATH} (validation loss {previous_val_loss:.5f})")
else:
    scaler = Standardizer().fit(x_tr)                      # training-set statistics only
    network = [
        Dense(X.shape[1], 16, init="he", l2=1e-4), ReLU(),
        Dense(16, 8, init="he", l2=1e-4), ReLU(),
        Dense(8, 1), Sigmoid(),
    ]
x_tr_s, x_va_s, x_te_s = (scaler.transform(a) for a in (x_tr, x_va, x_te))

# ---- 3. train, watching validation loss; keep the best epoch ----
history = train(network, binary_cross_entropy, binary_cross_entropy_prime,
                x_tr_s, y_tr, Adam(learning_rate=0.001),
                x_val=x_va_s, y_val=y_va, metric=accuracy,
                epochs=300, batch_size=64,
                patience=25, print_every=10, seed=0 if previous_val_loss is None else 1)

# When resuming, never replace a better saved model with a worse one
save = True
val_loss = binary_cross_entropy(y_va, predict(network, x_va_s))
if previous_val_loss is not None and val_loss >= previous_val_loss:
    print("Validation loss did not improve on the saved model, so the saved model is kept.")
    network, save = load_model(MODEL_PATH).network, False

# ---- 4. evaluate ONCE on the untouched test set ----
test_pred = predict(network, x_te_s)
test_acc = accuracy(y_te, test_pred)
print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Always-guess-majority baseline: {majority_baseline(y_te):.4f}")
cm = confusion_matrix(y_te, test_pred)
print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm)

# ---- 5. how much did it overfit? (skipped if a better saved model was kept) ----
if save:
    print()
    overfitting_report(history, test_score=test_acc, n_test=len(y_te))

# ---- 6. save the model together with its scaler ----
if save:
    path = save_model(MODEL_PATH, network, scaler, names,
                      metadata={"test_accuracy": test_acc,
                                "val_loss": float(binary_cross_entropy(y_va, predict(network, x_va_s)))})
    print(f"\nSaved model -> {path}")

# ---- 7. visualizations ----
plot_history(history).savefig("lol_training.png", dpi=150, bbox_inches="tight")
plot_confusion_matrix(cm, labels=("red won", "blue won")).savefig("lol_confusion.png", dpi=150, bbox_inches="tight")

rows = np.arange(6)                                       # first 6 test games
labels = [f"game {r + 1} ({'blue' if y_te[r, 0] else 'red'} won)" for r in rows]
plot_layer_flow(network, x_te_s[rows], y_te[rows], row_labels=labels, feature_names=names,
                title="Six test games through the network").savefig("lol_layers.png", dpi=150, bbox_inches="tight")

plot_probability_slice(network, scaler, x_te, y_te, names.index("goldDiff"), names.index("expDiff"),
                       names).savefig("lol_slice.png", dpi=150, bbox_inches="tight")
plt.show()
