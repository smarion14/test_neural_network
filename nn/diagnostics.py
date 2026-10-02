import numpy as np


def _gap_label(points):
    """Rough rule of thumb for the train-minus-validation accuracy gap."""
    if points < 3:
        return "small"
    if points <= 8:
        return "moderate"
    return "large"


def overfitting_report(history, test_score=None, n_test=None, metric_name="accuracy", verbose=True):
    """Summarize how much a network overfit, from the history dict returned by train().

    Needs validation data (x_val / y_val passed to train). Reports, at the best epoch:
      * train vs validation metric and loss
    and, if training continued past it:
      * how train and validation loss changed afterwards
    and, if you pass the final test score (and n_test, the number of test samples):
      * test vs validation, with the test score's ~95% sampling margin of error
        (this margin assumes the score is an accuracy, i.e. a fraction of samples correct)

    Returns the numbers as a dict. These labels are rules of thumb, not hard cutoffs.
    """
    if "val_loss" not in history or not history.get("best_epoch"):
        raise ValueError("history has no validation data. Pass x_val and y_val to train().")

    b = history["best_epoch"] - 1                     # index of the best epoch
    last = len(history["train_loss"]) - 1
    tl, vl = history["train_loss"], history["val_loss"]
    out = {"best_epoch": b + 1, "epochs_trained": last + 1,
           "train_loss": tl[b], "val_loss": vl[b], "loss_gap": vl[b] - tl[b]}
    lines = [f"Overfitting check at the best epoch ({b + 1} of {last + 1} trained)"]

    has_metric = "train_metric" in history and "val_metric" in history
    if has_metric:
        tm, vm = history["train_metric"][b], history["val_metric"][b]
        gap = (tm - vm) * 100
        out.update(train_metric=tm, val_metric=vm, metric_gap_points=gap, label=_gap_label(gap))
        lines.append(f"  {metric_name:<9} train {tm:.3f}   validation {vm:.3f}   "
                     f"gap {gap:+.1f} points -> {out['label']}")
    lines.append(f"  {'loss':<9} train {tl[b]:.3f}   validation {vl[b]:.3f}   gap {out['loss_gap']:+.3f}")

    after = last - b
    if after > 0:
        t_delta, v_delta = tl[last] - tl[b], vl[last] - vl[b]
        out.update(epochs_after_best=after, train_loss_change=t_delta, val_loss_change=v_delta)
        note = ""
        if t_delta < 0 and v_delta > -1e-3:
            note = "  -> training kept improving, validation did not"
        lines.append(f"  In the {after} epochs after the best one: train loss {t_delta:+.4f}, "
                     f"validation loss {v_delta:+.4f}{note}")

    if test_score is not None and has_metric:
        diff = (test_score - vm) * 100
        out["test_minus_val_points"] = diff
        text = f"  Test {metric_name} {test_score:.3f} vs validation {vm:.3f}: difference {diff:+.1f} points"
        if n_test:
            margin = 1.96 * np.sqrt(test_score * (1 - test_score) / n_test) * 100
            out["test_margin_points"] = margin
            verdict = ("within sampling noise" if abs(diff) <= margin
                       else "larger than sampling noise alone would usually explain")
            text += f" (95% margin of error on the test score: ±{margin:.1f} points), {verdict}"
        lines.append(text)

    lines.append("  Rough guide: accuracy gaps under 3 points are small, 3-8 moderate, over 8 large. "
                 "Accuracies on ~1,500 games are only good to about ±2 points (95%), so treat small gaps with caution.")
    if verbose:
        print("\n".join(lines))
    return out
