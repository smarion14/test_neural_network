import csv
import numpy as np

# Features are blue-minus-red differences wherever possible. The raw CSV stores
# both teams' stats, but many columns are mirrors of each other (redKills equals
# blueDeaths, redGoldDiff equals -blueGoldDiff), so differences remove the
# redundancy and give the network fewer, more meaningful inputs.
LOL_FEATURES = [
    ("goldDiff",        lambda d: d["blueGoldDiff"]),
    ("expDiff",         lambda d: d["blueExperienceDiff"]),
    ("killDiff",        lambda d: d["blueKills"] - d["redKills"]),
    ("assistDiff",      lambda d: d["blueAssists"] - d["redAssists"]),
    ("firstBlood",      lambda d: d["blueFirstBlood"]),
    ("towerDiff",       lambda d: d["blueTowersDestroyed"] - d["redTowersDestroyed"]),
    ("dragonDiff",      lambda d: d["blueDragons"] - d["redDragons"]),
    ("heraldDiff",      lambda d: d["blueHeralds"] - d["redHeralds"]),
    ("csDiff",          lambda d: d["blueTotalMinionsKilled"] - d["redTotalMinionsKilled"]),
    ("jungleCsDiff",    lambda d: d["blueTotalJungleMinionsKilled"] - d["redTotalJungleMinionsKilled"]),
    ("levelDiff",       lambda d: d["blueAvgLevel"] - d["redAvgLevel"]),
    ("wardsPlacedDiff", lambda d: d["blueWardsPlaced"] - d["redWardsPlaced"]),
    ("wardsKilledDiff", lambda d: d["blueWardsDestroyed"] - d["redWardsDestroyed"]),
]

REQUIRED_COLUMNS = [
    "blueWins", "blueGoldDiff", "blueExperienceDiff", "blueKills", "redKills",
    "blueAssists", "redAssists", "blueFirstBlood", "blueTowersDestroyed",
    "redTowersDestroyed", "blueDragons", "redDragons", "blueHeralds", "redHeralds",
    "blueTotalMinionsKilled", "redTotalMinionsKilled", "blueTotalJungleMinionsKilled",
    "redTotalJungleMinionsKilled", "blueAvgLevel", "redAvgLevel", "blueWardsPlaced",
    "redWardsPlaced", "blueWardsDestroyed", "redWardsDestroyed",
]


def load_lol(path):
    """Load high_diamond_ranked_10min.csv.

    Returns X (games, features), y (games, 1) where 1 means blue won,
    and the list of feature names. Uses only the standard library + NumPy.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            columns = {name: [] for name in reader.fieldnames}
            for row in reader:
                for name, value in row.items():
                    columns[name].append(float(value))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{path}'. Download high_diamond_ranked_10min.csv from Kaggle "
            "('League of Legends Diamond Ranked Games (10 min)') and put it in the data/ folder."
        ) from None

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    data = {name: np.array(values) for name, values in columns.items()}
    names = [name for name, _ in LOL_FEATURES]
    X = np.column_stack([fn(data) for _, fn in LOL_FEATURES])
    y = data["blueWins"].reshape(-1, 1)
    return X, y, names


def train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, seed=0):
    """Shuffle once, then cut into train / validation / test."""
    idx = np.random.default_rng(seed).permutation(len(X))
    n_test, n_val = int(len(X) * test_frac), int(len(X) * val_frac)
    test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


class Standardizer:
    """Scale each feature to mean 0, std 1.

    Fit on the TRAINING set only, then reuse the same mean/std on validation and
    test data. Fitting on all the data would leak test-set information into training.
    """

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0          # constant column: avoid divide-by-zero
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
