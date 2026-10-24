"""Swap the two teams and every blue-minus-red feature changes sign.

A model should not care which side is called blue: p(blue wins | x) = 1 - p(blue wins | mirror(x)).
Training on both views doubles the rows without storing another game, and averaging a prediction with
its mirror's makes the answer exactly independent of which team was called blue.

`signs` says how each feature behaves under the swap: -1 for blue-minus-red features, +1 for any
that are the same for both teams (such as the tier of the game).
"""
import numpy as np


def mirror(X, signs):
    """The same games seen from the other side."""
    return X * signs


def augment(X, y, signs):
    """Each game plus its mirror image. The mirror's label flips: blue won becomes red won."""
    return np.concatenate([X, mirror(X, signs)]), np.concatenate([y, 1 - y])


def symmetric(predict, X, signs):
    """predict(X) averaged with the mirror's answer, so the result cannot depend on which team was called blue."""
    return 0.5 * (predict(X) + 1 - predict(mirror(X, signs)))
