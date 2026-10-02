import json
import os
from collections import namedtuple

import numpy as np

from .dense import Dense
from .activations import Tanh, Sigmoid, ReLU
from .data import Standardizer

FORMAT_VERSION = 1
ACTIVATIONS = {"Tanh": Tanh, "Sigmoid": Sigmoid, "ReLU": ReLU}

LoadedModel = namedtuple("LoadedModel", "network scaler feature_names metadata")


def save_model(path, network, scaler=None, feature_names=None, metadata=None):
    spec, arrays = [], {}
    for i, layer in enumerate(network):
        name = type(layer).__name__
        if isinstance(layer, Dense):
            spec.append({"type": "Dense", "l2": layer.l2})
            arrays[f"layer{i}_weights"] = layer.weights
            arrays[f"layer{i}_bias"] = layer.bias
        elif name in ACTIVATIONS:
            spec.append({"type": name})
        else:
            raise ValueError(f"Cannot save layer type '{name}'. Supported: Dense, "
                             f"{', '.join(ACTIVATIONS)}.")

    header = {"format_version": FORMAT_VERSION, "layers": spec,
              "feature_names": list(feature_names) if feature_names is not None else None,
              "metadata": metadata or {}}
    arrays["header"] = np.array(json.dumps(header))
    if scaler is not None:
        arrays["scaler_mean"], arrays["scaler_std"] = scaler.mean, scaler.std

    if not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_model(path):
    if not path.endswith(".npz"):
        path += ".npz"
    try:
        data = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"No saved model at '{path}'.") from None

    header = json.loads(str(data["header"]))
    if header["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported model file version {header['format_version']}.")

    network = []
    for i, spec in enumerate(header["layers"]):
        if spec["type"] == "Dense":
            w, b = data[f"layer{i}_weights"], data[f"layer{i}_bias"]
            layer = Dense(w.shape[0], w.shape[1], l2=spec["l2"])
            layer.weights, layer.bias = w.copy(), b.copy()
            network.append(layer)
        else:
            network.append(ACTIVATIONS[spec["type"]]())

    scaler = None
    if "scaler_mean" in data:
        scaler = Standardizer()
        scaler.mean, scaler.std = data["scaler_mean"].copy(), data["scaler_std"].copy()
    return LoadedModel(network, scaler, header["feature_names"], header["metadata"])
