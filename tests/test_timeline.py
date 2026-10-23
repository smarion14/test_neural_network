"""Collector parsing, timeline features, and the boosted-tree model."""
import numpy as np

from collect.riot import compact
from nn import GradientBoosting, ChampionPrior, build_features, accuracy


def fake_match(win_blue=True):
    parts = [{"participantId": i + 1, "teamId": 100 if i < 5 else 200, "championId": 10 + i,
              "teamPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][i % 5],
              "win": win_blue == (i < 5)} for i in range(10)]
    return {"metadata": {"matchId": "NA1_1"},
            "info": {"gameDuration": 1500, "gameVersion": "14.5.123.456", "participants": parts}}


def fake_timeline(minutes=8):
    frames = []
    for m in range(minutes):
        pf = {str(p): {"totalGold": 500 * m + (100 * m if p <= 5 else 0), "xp": 300 * m,
                       "minionsKilled": 7 * m, "jungleMinionsKilled": 0, "level": 1 + m // 2}
              for p in range(1, 11)}
        frames.append({"participantFrames": pf, "events": []})
    frames[3]["events"] = [
        {"type": "CHAMPION_KILL", "timestamp": 200000, "victimId": 7, "killerId": 1,
         "assistingParticipantIds": [2, 3]},                       # blue kills red 7, two assists
        {"type": "ELITE_MONSTER_KILL", "timestamp": 210000, "monsterType": "DRAGON", "killerTeamId": 100},
        {"type": "BUILDING_KILL", "timestamp": 220000, "buildingType": "TOWER_BUILDING", "teamId": 100},
        {"type": "WARD_PLACED", "timestamp": 230000, "creatorId": 6}]
    return {"info": {"frames": frames}}


def games():
    out = []
    for i in range(6):
        g = compact(fake_match(win_blue=i % 2 == 0), fake_timeline(), "GOLD")
        g["champs"] = [c + i for c in g["champs"]]
        out.append(g)
    return out


def test_compact_skips_short_games():
    m = fake_match()
    m["info"]["gameDuration"] = 200
    assert compact(m, fake_timeline(), "GOLD") is None


def test_features_match_hand_computed_values():
    g = games()
    X, y, names = build_features(g, 5, ChampionPrior().fit(g))
    row = dict(zip(names, X[0]))
    assert row["goldDiff"] == 5 * 100 * 5                 # blue +100/min each, 5 players, 5 min
    assert row["killDiff"] == 1 and row["assistDiff"] == 2 and row["firstBlood"] == 1
    assert row["dragonDiff"] == 1                         # blue took the dragon
    assert row["towerDiff"] == -1                         # the tower that fell was blue's
    assert row["wardsPlacedDiff"] == -1                   # player 6 is red
    assert row["tier"] == 3 and y[0, 0] == 1 and y[1, 0] == 0


def test_features_ignore_the_future():
    g = games()
    X, _, names = build_features(g, 2, ChampionPrior().fit(g))   # the kill happens in minute 3
    assert X[0, names.index("killDiff")] == 0


def test_boosting_learns_an_interaction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)    # XOR of two features
    model = GradientBoosting(depth=3).fit(X[:1500], y[:1500], X[1500:], y[1500:])
    assert accuracy(y[1500:], model.predict(X[1500:])) > 0.95


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
