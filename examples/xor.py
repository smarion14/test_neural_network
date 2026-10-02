"""Solve XOR and visualise what the network does.

Run from the project root:   python -m examples.xor
"""
import numpy as np
import matplotlib.pyplot as plt

from nn import (Dense, Tanh, Sigmoid, binary_cross_entropy,
                binary_cross_entropy_prime, predict, train, SGD)
from nn.visualize import plot_loss, plot_layer_flow, plot_decision_boundary

np.random.seed(1)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

network = [
    Dense(2, 4), Tanh(),
    Dense(4, 1), Sigmoid(),
]

# Same network, same inputs: before vs after training
plot_layer_flow(network, X, Y, title="BEFORE training").savefig("xor_layers_before.png", dpi=150, bbox_inches="tight")

history = train(network, binary_cross_entropy, binary_cross_entropy_prime,
                X, Y, SGD(learning_rate=0.5), epochs=2000, print_every=250)

print("\nPredictions after training:")
for x, y in zip(X, Y):
    print(f"  {x.astype(int)} -> {predict(network, x[None, :])[0, 0]:.4f}   (target {int(y[0])})")

plot_layer_flow(network, X, Y, title="AFTER training").savefig("xor_layers_after.png", dpi=150, bbox_inches="tight")
plot_loss(history, "XOR: loss falls as the network learns").savefig("xor_loss.png", dpi=150, bbox_inches="tight")
plot_decision_boundary(network, X, Y, title="XOR: output over the input space").savefig("xor_boundary.png", dpi=150, bbox_inches="tight")

plt.show()
