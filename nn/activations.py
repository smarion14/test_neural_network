import numpy as np
from .layer import Layer


class Activation(Layer):
    """Generic activation layer: wraps a function f and its derivative f'."""

    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_gradient):
        # Elementwise chain rule: dE/dX = dE/dY * f'(X). No parameters to store.
        return output_gradient * self.activation_prime(self.input)


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            # clip so exp() can't overflow for very large |x|
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
