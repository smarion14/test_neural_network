import json

import numpy as np

TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

# frames[minute][player] = [gold, xp, minions, jungle minions, level]; players 0-4 are
# blue, 5-9 red. Events are [kind, minute, who]: "kill" (who = victim player, plus the
# number of assists), "dragon" / "herald" / "tower" (who = team that took it, 0 blue /
# 1 red), "ward" / "wardkill" (who = player).

FEATURES = ["goldDiff", "goldDiffLast3", "expDiff", "expDiffLast3", "csDiff", "jungleCsDiff",
            "levelDiff", "topGoldGap", "killDiff", "killDiffLast3", "assistDiff", "firstBlood",
            "towerDiff", "dragonDiff", "heraldDiff", "wardsPlacedDiff", "wardsKilledDiff",
            *[f"{r.lower()}GoldDiff" for r in ROLES], "champPriorDiff", "tier"]
# If blue and red swap (see nn/symmetry.py) every feature changes sign except the tier, which both share.
MIRROR_SIGNS = np.array([1.0 if name == "tier" else -1.0 for name in FEATURES])


def load_games(path):
    """Read the games written by collect/riot.py (one JSON game per line)."""
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"No games at '{path}'. Run: python -m collect.riot") from None


class ChampionPrior:
    """Smoothed win rate of each champion, learned from the TRAINING games only.

    Fit it on the training split and reuse it on validation and test, exactly like
    the Standardizer, or the test games' outcomes leak into their own features.
    The training games leak too: their own result is inside the rates they are scored
    with, so the feature looks far more predictive on them than it will on new games.
    Score those with team_diff(game, own_game=True), which takes the game's own result out.
    """

    def __init__(self, strength=20):
        self.strength = strength

    @staticmethod
    def _won(game, p):
        return game["blueWin"] if p < 5 else 1 - game["blueWin"]

    def fit(self, games):
        self.wins, self.seen = {}, {}
        for g in games:
            for p, champ in enumerate(g["champs"]):
                self.wins[champ] = self.wins.get(champ, 0) + self._won(g, p)
                self.seen[champ] = self.seen.get(champ, 0) + 1
        return self

    def team_diff(self, game, own_game=False):
        """Blue minus red mean champion rate. own_game=True: `game` was in fit(); leave it out."""
        k, rates = self.strength, []
        for p, champ in enumerate(game["champs"]):
            wins, seen = self.wins.get(champ, 0), self.seen.get(champ, 0)
            if own_game:
                wins, seen = wins - self._won(game, p), seen - 1
            rates.append((wins + 0.5 * k) / (seen + k) if seen + k else 0.5)
        return np.mean(rates[:5]) - np.mean(rates[5:])


def _by_role(values, roles):
    """Order one team's 5 values as top, jungle, mid, bottom, support (positional if roles are unclear)."""
    if sorted(roles) != sorted(ROLES):
        return values
    return values[[roles.index(r) for r in ROLES]]


def build_features(games, minute, prior, training=False):
    """Blue-minus-red features at `minute`. Every game must have more than `minute` frames.

    training=True for the games `prior` was fitted on (see ChampionPrior).
    Returns X (games, features), y (games, 1) where 1 means blue won, and the names.
    """
    rows = []
    for g in games:
        F = np.array(g["frames"][:minute + 1], float)             # (minute+1, 10, 5)
        blue, red = F[:, :5], F[:, 5:]
        gold = blue[:, :, 0].sum(1) - red[:, :, 0].sum(1)         # per minute
        exp = blue[:, :, 1].sum(1) - red[:, :, 1].sum(1)
        past = max(minute - 3, 0)
        role_gold = (_by_role(blue[-1, :, 0], g["roles"][:5])
                     - _by_role(red[-1, :, 0], g["roles"][5:]))

        kills, assists, first = np.zeros(2), np.zeros(2), 0
        recent_kills = np.zeros(2)
        count = {"tower": np.zeros(2), "dragon": np.zeros(2), "herald": np.zeros(2),
                 "ward": np.zeros(2), "wardkill": np.zeros(2)}
        for kind, t, who, *extra in sorted(g["events"], key=lambda e: e[1]):
            if t > minute:
                break
            if kind == "kill":
                killer_team = 0 if who >= 5 else 1                 # the victim's opponents
                kills[killer_team] += 1
                assists[killer_team] += extra[0]
                recent_kills[killer_team] += t > minute - 3
                first = first or (1 if killer_team == 0 else -1)
            elif kind in ("ward", "wardkill"):
                count[kind][0 if who < 5 else 1] += 1
            else:
                count[kind][who] += 1

        rows.append([gold[-1], gold[-1] - gold[past], exp[-1], exp[-1] - exp[past],
                     blue[-1, :, 2].sum() - red[-1, :, 2].sum(),
                     blue[-1, :, 3].sum() - red[-1, :, 3].sum(),
                     blue[-1, :, 4].mean() - red[-1, :, 4].mean(),
                     blue[-1, :, 0].max() - red[-1, :, 0].max(),
                     kills[0] - kills[1], recent_kills[0] - recent_kills[1],
                     assists[0] - assists[1], first,
                     *(count[k][0] - count[k][1] for k in ("tower", "dragon", "herald", "ward", "wardkill")),
                     *role_gold, prior.team_diff(g, own_game=training), TIERS.index(g["tier"])])
    y = np.array([[g["blueWin"]] for g in games], float)
    return np.array(rows), y, list(FEATURES)


def build_rows(games, minutes, prior, training=False):
    """build_features at several minutes at once: one row per game and minute, for the games that lasted that long.

    The minute is added as the last column. Returns X, y, the position in `games` of each row's game, and the names.
    """
    X, y, which = [], [], []
    for minute in minutes:
        lasted = [i for i, g in enumerate(games) if len(g["frames"]) > minute]
        if lasted:
            rows, labels, _ = build_features([games[i] for i in lasted], minute, prior, training)
            X.append(np.column_stack([rows, np.full(len(lasted), minute)]))
            y.append(labels)
            which.append(lasted)
    return np.vstack(X), np.vstack(y), np.concatenate(which), [*FEATURES, "minute"]
