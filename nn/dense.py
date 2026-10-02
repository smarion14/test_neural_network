import numpy as np
from .layer import Layer


class Dense(Layer):
    """Fully connected layer: Y = XW + B.

    init : "xavier" (scale sqrt(1/n_in), good for tanh/sigmoid)
           or "he"  (scale sqrt(2/n_in), good for ReLU)
    l2   : L2 regularization strength. Adds (l2/2) * sum(W^2) to the loss, which
           contributes l2 * W to dE/dW. Biases are not penalized. The penalty is
           NOT included in the loss value that train() reports.
    """

    def __init__(self, input_size, output_size, init="xavier", l2=0.0):
        super().__init__()
        scales = {"xavier": np.sqrt(1.0 / input_size), "he": np.sqrt(2.0 / input_size)}
        if init not in scales:
            raise ValueError(f"init must be one of {list(scales)}, got '{init}'")
        self.weights = np.random.randn(input_size, output_size) * scales[init]
        self.bias = np.zeros((1, output_size))
        self.l2 = l2
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input                                  # (batch, input_size)
        self.output = input @ self.weights + self.bias      # Y = XW + B
        return self.output

    def backward(self, output_gradient):
        # output_gradient is dE/dY, shape (batch, output_size)
        self.weights_grad = self.input.T @ output_gradient + self.l2 * self.weights
        self.bias_grad = np.sum(output_gradient, axis=0, keepdims=True)
        return output_gradient @ self.weights.T             # dE/dX, for the previous layer

    def params(self):
        return [self.weights, self.bias]

    def grads(self):
        return [self.weights_grad, self.bias_grad]
