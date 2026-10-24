"""Collector parsing, timeline features, and the boosted-tree model."""
import numpy as np

from collect.riot import compact
from nn import GradientBoosting, ChampionPrior, MIRROR_SIGNS, build_features, build_rows, accuracy


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


def test_champion_prior_leaves_the_games_own_result_out():
    champs = list(range(10))
    blue_won, red_won = {"champs": champs, "blueWin": 1}, {"champs": champs, "blueWin": 0}
    prior = ChampionPrior(strength=1).fit([blue_won, red_won])
    assert prior.team_diff(blue_won) == 0                  # both results counted: every champion looks 50/50
    # without blue_won's own result the blue champions are 0.25 and the red ones 0.75
    assert np.isclose(prior.team_diff(blue_won, own_game=True), -0.5)


def test_only_training_games_get_the_leave_one_out_prior():
    g = games()
    prior = ChampionPrior().fit(g)
    plain, _, names = build_features(g, 5, prior)
    loo, _, _ = build_features(g, 5, prior, training=True)
    c = names.index("champPriorDiff")
    assert np.allclose(plain[:, c], [prior.team_diff(x) for x in g])
    assert np.allclose(loo[:, c], [prior.team_diff(x, own_game=True) for x in g])
    assert not np.allclose(plain[:, c], loo[:, c])


def test_build_rows_stacks_the_minutes_and_skips_games_that_have_ended():
    g = games() + [compact(fake_match(), fake_timeline(minutes=6), "GOLD")]     # the last game ends after minute 5
    prior = ChampionPrior().fit(g)
    X, y, which, names = build_rows(g, [5, 6], prior)
    assert names[-1] == "minute" and X.shape == (7 + 6, len(names))             # 7 games at minute 5, 6 at minute 6
    assert X[:7, -1].tolist() == [5] * 7 and X[7:, -1].tolist() == [6] * 6
    assert which.tolist() == list(range(7)) + list(range(6))
    assert np.array_equal(X[:7, :-1], build_features(g, 5, prior)[0])


def swap_teams(g):
    """The same game with blue and red exchanged."""
    flip = lambda who: (who + 5) % 10                              # player index on the other team
    events = [[kind, t, flip(who) if kind in ("kill", "ward", "wardkill") else 1 - who, *extra]
              for kind, t, who, *extra in g["events"]]            # towers, dragons, heralds: who is a team
    return {**g, "frames": [f[5:] + f[:5] for f in g["frames"]], "events": events,
            "champs": g["champs"][5:] + g["champs"][:5], "roles": g["roles"][5:] + g["roles"][:5],
            "blueWin": 1 - g["blueWin"]}


def test_swapping_the_teams_flips_exactly_the_features_that_mirror_signs_says():
    g = games()
    prior = ChampionPrior().fit(g)
    X, y, names = build_features(g, 5, prior)
    X_swapped, y_swapped, _ = build_features([swap_teams(x) for x in g], 5, prior)
    assert np.allclose(X_swapped, X * MIRROR_SIGNS)
    assert np.array_equal(y_swapped, 1 - y)
    assert np.count_nonzero(X[0, MIRROR_SIGNS < 0]) >= 10          # not vacuous: most of the features are nonzero here
    assert [n for n, s in zip(names, MIRROR_SIGNS) if s > 0] == ["tier"]


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
