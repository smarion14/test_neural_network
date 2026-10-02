import numpy as np


class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    """Gradient descent, optionally with momentum.

    Plain:     p <- p - lr * g
    Momentum:  v <- m * v - lr * g ;  p <- p + v
    """

    def __init__(self, learning_rate=0.1, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.momentum == 0.0:
            for p, g in zip(params, grads):
                p -= self.lr * g
            return
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam(Optimizer):
    """Adam: per-parameter step sizes from running averages of the gradient (m)
    and its square (v), with bias correction for the early steps."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
