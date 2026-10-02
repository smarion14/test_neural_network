class Layer:
    """Base class. A layer computes outputs, and gradients; it never updates itself.

    Updating parameters is the optimizer's job (see optimizers.py). Keeping the two
    apart means any layer can be trained with any optimizer, and a layer's gradients
    can be checked numerically without the weights changing underneath you.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """Take input X, return output Y."""
        raise NotImplementedError

    def backward(self, output_gradient):
        """Take dE/dY. Store dE/d(parameters) on the layer. Return dE/dX."""
        raise NotImplementedError

    def params(self):
        """Trainable arrays (updated in place by the optimizer). None by default."""
        return []

    def grads(self):
        """Gradients matching params(), in the same order. Valid after backward()."""
        return []
