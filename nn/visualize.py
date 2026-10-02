import numpy as np
import matplotlib.pyplot as plt
from .network import predict, trace

BLUE, ORANGE = "#4C72B0", "#DD8452"   # colorblind-friendly pair


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_loss(history, title="Training loss"):
    if not isinstance(history, dict):
        history = {"train_loss": history}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], color=BLUE, linewidth=2, label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], color=ORANGE, linewidth=2, linestyle="--", label="validation")
        ax.legend(frameon=False)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title, fontweight="bold")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_history(history, metric_name="Accuracy", title="Training vs validation"):
    fig, (ax_l, ax_m) = plt.subplots(1, 2, figsize=(11, 4))
    panels = [(ax_l, "loss", "Loss"), (ax_m, "metric", metric_name)]
    for ax, key, label in panels:
        if f"train_{key}" not in history:
            ax.set_visible(False)
            continue
        ax.plot(history[f"train_{key}"], color=BLUE, linewidth=2, label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], color=ORANGE, linewidth=2, linestyle="--", label="validation")
        if history.get("best_epoch"):
            ax.axvline(history["best_epoch"] - 1, color="grey", linestyle=":", linewidth=1.5)
            ax.text(history["best_epoch"] - 1, ax.get_ylim()[0], " best epoch", color="grey",
                    va="bottom", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.legend(frameon=False)
        _clean(ax)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels=("0", "1"), title="Confusion matrix (test set)"):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    row_totals = cm.sum(axis=1, keepdims=True)
    for r in range(2):
        for c in range(2):
            dark = cm[r, c] > cm.max() * 0.55
            ax.text(c, r, f"{cm[r, c]}\n({cm[r, c] / row_totals[r, 0]:.0%} of row)",
                    ha="center", va="center", color="white" if dark else "black", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"predicted\n{l}" for l in labels])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"actual\n{l}" for l in labels])
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_layer_flow(network, x, y_true=None, row_labels=None, feature_names=None,
                    annotate=None, title="Layer-by-layer forward pass"):
    steps = trace(network, x)
    if y_true is not None:
        steps.append(("Target", y_true))

    n_rows = len(x)
    if row_labels is None:
        row_labels = [str(np.round(r, 2).tolist()) for r in x]
    if annotate is None:
        annotate = sum(a.shape[1] for _, a in steps) <= 24

    widths = [max(a.shape[1], 1) + 0.6 for _, a in steps]
    fig, axes = plt.subplots(
        1, len(steps), figsize=(min(18, sum(widths) * 0.9 + 1.5), 0.75 * n_rows + 1.8),
        gridspec_kw={"width_ratios": widths},
    )
    for i, (ax, (label, a)) in enumerate(zip(axes, steps)):
        vmax = max(np.abs(a).max(), 1e-9)               # symmetric scale: 0 = white
        ax.imshow(a, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        if annotate:
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    strong = abs(a[r, c]) > 0.6 * vmax
                    ax.text(c, r, f"{a[r, c]:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if strong else "black")
        narrow = a.shape[1] <= 2                       # wrap titles on slim panels so they don't collide
        ax.set_title(label.replace(" ", "\n") if narrow else label,
                     fontsize=8 if narrow else 10, fontweight="bold")
        ax.set_xticks(range(a.shape[1]))
        if i == 0 and feature_names is not None:
            ax.set_xticklabels(feature_names, fontsize=7, rotation=90)
        else:
            ax.set_xticklabels([f"n{c + 1}" for c in range(a.shape[1])], fontsize=8,
                               rotation=90 if a.shape[1] > 12 else 0)
        ax.set_yticks(range(n_rows) if i == 0 else [])
        if i == 0:
            ax.set_yticklabels(row_labels, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(title + "   (blue = negative, red = positive)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_decision_boundary(network, x, y, resolution=250, pad=0.4, title="Network output over the input space"):
    x1 = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, resolution)
    x2 = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, resolution)
    xx, yy = np.meshgrid(x1, x2)
    zz = predict(network, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cs = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.85)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5)
    fig.colorbar(cs, ax=ax, label="Network output")

    labels = y.ravel() > 0.5
    ax.scatter(x[~labels, 0], x[~labels, 1], c=BLUE, s=130, edgecolor="black", marker="o", label="target 0", zorder=3)
    ax.scatter(x[labels, 0], x[labels, 1], c=ORANGE, s=130, edgecolor="black", marker="s", label="target 1", zorder=3)
    ax.set_xlabel("input 1")
    ax.set_ylabel("input 2")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper center", ncol=2, frameon=True)
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_probability_slice(network, scaler, X_raw, y, i, j, feature_names,
                           resolution=200, n_points=300, seed=0,
                           title="Predicted chance of a blue win"):
    lo_i, hi_i = np.percentile(X_raw[:, i], [1, 99])
    lo_j, hi_j = np.percentile(X_raw[:, j], [1, 99])
    gx, gy = np.meshgrid(np.linspace(lo_i, hi_i, resolution), np.linspace(lo_j, hi_j, resolution))
    grid = np.tile(X_raw.mean(axis=0), (gx.size, 1))
    grid[:, i], grid[:, j] = gx.ravel(), gy.ravel()
    z = predict(network, scaler.transform(grid)).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    cs = ax.contourf(gx, gy, z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.85)
    ax.contour(gx, gy, z, levels=[0.5], colors="black", linewidths=1.5)
    fig.colorbar(cs, ax=ax, label="P(blue wins)")

    pick = np.random.default_rng(seed).choice(len(X_raw), size=min(n_points, len(X_raw)), replace=False)
    won = y[pick].ravel() > 0.5
    ax.scatter(X_raw[pick][~won, i], X_raw[pick][~won, j], c=BLUE, s=22, edgecolor="white",
               linewidth=0.5, marker="o", label="red won", zorder=3)
    ax.scatter(X_raw[pick][won, i], X_raw[pick][won, j], c=ORANGE, s=26, edgecolor="white",
               linewidth=0.5, marker="s", label="blue won", zorder=3)
    ax.set_xlim(lo_i, hi_i); ax.set_ylim(lo_j, hi_j)
    ax.set_xlabel(feature_names[i]); ax.set_ylabel(feature_names[j])
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_model_comparison(scores, baseline=None, ylabel="Test accuracy",
                          title="Same splits, different models"):
    names = list(scores)
    data = np.array([scores[k] for k in names], dtype=float)          # (models, splits)
    colors, markers = [BLUE, ORANGE, "#55A868"], ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for s in range(data.shape[1]):
        ax.plot(range(len(names)), data[:, s], color="lightgrey", linewidth=1, zorder=1)
    for i, name in enumerate(names):
        ax.scatter(np.full(data.shape[1], i), data[i], s=55, color=colors[i % 3],
                   marker=markers[i % 3], edgecolor="black", linewidth=0.5, zorder=3)
        ax.hlines(data[i].mean(), i - 0.22, i + 0.22, color="black", linewidth=2, zorder=4)
        ax.text(i + 0.26, data[i].mean(), f"mean {data[i].mean():.3f}", va="center", fontsize=9)
    if baseline is not None:
        ax.axhline(np.mean(baseline), color="grey", linestyle="--", linewidth=1.2)
        ax.text(len(names) - 0.55, np.mean(baseline), "always guess majority", color="grey",
                fontsize=9, va="bottom", ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.5, len(names) - 0.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    _clean(ax)
    fig.tight_layout()
    return fig
