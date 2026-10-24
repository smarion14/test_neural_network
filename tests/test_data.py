"""The data pipeline: loading the CSV, splitting it, and scaling without peeking at test data."""
import os
import tempfile

import numpy as np

from nn import Standardizer, load_lol, train_val_test_split
from nn.data import LOL_FEATURES, REQUIRED_COLUMNS


def write_csv(path, rows, drop=None):
    columns = [c for c in REQUIRED_COLUMNS if c != drop]
    with open(path, "w", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, 0)) for c in columns) + "\n")


def test_split_is_disjoint_complete_and_keeps_labels_with_their_rows():
    X = np.arange(1000, dtype=float).reshape(-1, 1)             # each row's value is its own id
    parts = train_val_test_split(X, X.copy(), val_frac=0.15, test_frac=0.15, seed=3)
    ids = np.concatenate([x for x, _ in parts]).ravel()
    assert len(ids) == 1000 and len(set(ids)) == 1000           # nothing lost, nothing in two sets
    assert [len(x) for x, _ in parts] == [700, 150, 150]
    assert all(np.array_equal(x, y) for x, y in parts)


def test_standardizer_uses_training_statistics_only():
    train = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])      # the second column is constant
    scaler = Standardizer().fit(train)
    z = scaler.transform(train)
    assert np.allclose(z.mean(axis=0), 0) and np.isclose(z[:, 0].std(), 1)
    assert np.all(z[:, 1] == 0)                                 # constant column: no divide by zero
    far = scaler.transform(np.array([[102.0, 5.0]]))            # a test game far from the training data
    assert np.isclose(far[0, 0], (102 - 2) / train[:, 0].std())  # scaled with the training mean, not re-centred


def test_load_lol_builds_blue_minus_red_features():
    rows = [{"blueWins": 1, "blueGoldDiff": 500, "blueKills": 5, "redKills": 2, "blueFirstBlood": 1},
            {"blueWins": 0, "blueGoldDiff": -300, "blueKills": 1, "redKills": 4, "blueFirstBlood": 0}]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "games.csv")
        write_csv(path, rows)
        X, y, names = load_lol(path)
    assert names == [n for n, _ in LOL_FEATURES] and X.shape == (2, len(names))
    first = dict(zip(names, X[0]))
    assert first["goldDiff"] == 500 and first["killDiff"] == 3 and first["firstBlood"] == 1
    second = dict(zip(names, X[1]))
    assert second["killDiff"] == -3 and second["firstBlood"] == -1           # red got first blood
    assert y.tolist() == [[1.0], [0.0]]


def test_load_lol_explains_a_missing_column_and_a_missing_file():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "games.csv")
        write_csv(path, [{}], drop="redKills")
        for call, error, word in [(path, ValueError, "redKills"), (os.path.join(folder, "nope.csv"), FileNotFoundError, "Kaggle")]:
            try:
                load_lol(call)
            except error as e:
                assert word in str(e)
            else:
                raise AssertionError(f"expected {error.__name__}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
