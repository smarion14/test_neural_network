import numpy as np


def predict(network, x):
    """Forward pass through a list of layers."""
    output = x
    for layer in network:
        output = layer.forward(output)
    return output


def trace(network, x):
    """Forward pass that records every layer's output, for visualization."""
    steps = [("Input", x)]
    output = x
    for layer in network:
        output = layer.forward(output)
        if hasattr(layer, "weights"):
            n_in, n_out = layer.weights.shape
            label = f"Dense {n_in}→{n_out}"
        else:
            label = type(layer).__name__
        steps.append((label, output))
    return steps


def _get_params(network):
    return [[p.copy() for p in layer.params()] for layer in network]


def _set_params(network, saved):
    for layer, arrays in zip(network, saved):
        for p, a in zip(layer.params(), arrays):
            p[...] = a                                # in place, so the optimizer's state stays valid


def train(network, loss, loss_prime, x_train, y_train, optimizer,
          epochs=1000, batch_size=None,
          x_val=None, y_val=None, metric=None, patience=None,
          verbose=True, print_every=100, seed=None):
    """Train with mini-batch gradient descent, using the given optimizer.

    optimizer : an Optimizer from nn.optimizers (SGD, Adam, ...)

    Optional extras:
      x_val, y_val : held-out data, evaluated after every epoch (never trained on)
      metric       : function(y_true, y_pred) -> float, e.g. accuracy
      patience     : stop if validation loss hasn't improved for this many epochs,
                     then restore the weights from the best epoch

    Returns a history dict with train_loss (and val_loss, train_metric,
    val_metric when available), plus best_epoch / stopped_epoch.
    """
    n = len(x_train)
    batch_size = batch_size or n                      # None -> full batch
    rng = np.random.default_rng(seed)
    use_val = x_val is not None and y_val is not None
    name = getattr(metric, "__name__", "metric")

    history = {"train_loss": []}
    if use_val:
        history["val_loss"] = []
    if metric:
        history["train_metric"] = []
        if use_val:
            history["val_metric"] = []

    best_val, best_params, best_epoch, wait = np.inf, None, None, 0
    params = [p for layer in network for p in layer.params()]   # same arrays every step

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)                    # reshuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            output = predict(network, x_train[idx])                   # 1. forward
            grad = loss_prime(y_train[idx], output)                   # 2. error gradient
            for layer in reversed(network):                           # 3. backward: gradients only
                grad = layer.backward(grad)
            optimizer.step(params, [g for layer in network for g in layer.grads()])  # 4. update

        # Evaluate on full train / validation sets after the epoch
        train_pred = predict(network, x_train)
        history["train_loss"].append(loss(y_train, train_pred))
        if metric:
            history["train_metric"].append(metric(y_train, train_pred))
        if use_val:
            val_pred = predict(network, x_val)
            val_loss = loss(y_val, val_pred)
            history["val_loss"].append(val_loss)
            if metric:
                history["val_metric"].append(metric(y_val, val_pred))
            if val_loss < best_val - 1e-9:
                best_val, best_epoch, wait = val_loss, epoch, 0
                best_params = _get_params(network)
            else:
                wait += 1

        if verbose and epoch % print_every == 0:
            msg = f"epoch {epoch:>4}/{epochs}   loss {history['train_loss'][-1]:.5f}"
            if metric:
                msg += f"   {name} {history['train_metric'][-1]:.4f}"
            if use_val:
                msg += f"   | val loss {history['val_loss'][-1]:.5f}"
                if metric:
                    msg += f"   val {name} {history['val_metric'][-1]:.4f}"
            print(msg)

        if use_val and patience and wait >= patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    history["stopped_epoch"] = epoch
    history["best_epoch"] = best_epoch
    if use_val and patience and best_params is not None:
        _set_params(network, best_params)             # roll back to the best epoch
        if verbose:
            print(f"restored weights from best epoch {best_epoch} (val loss {best_val:.5f})")
    return history
