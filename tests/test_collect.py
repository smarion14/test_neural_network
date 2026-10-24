"""Storing and reading collected games: plain text and gzip."""
import os
import tempfile

from collect.pack import pack
from collect.riot import write_game
from nn import load_games


def fake_games(n=3):
    return [{"id": f"NA1_{i}", "tier": "GOLD", "blueWin": i % 2, "frames": [[[1, 2, 3, 4, 5]] * 10] * 8, "events": []}
            for i in range(n)]


def write(path, games, packed):
    with open(path, "ab") as out:                       # as the collector does, one game at a time
        for g in games:
            write_game(out, g, packed)


def test_games_come_back_the_same_from_plain_and_gzip_files():
    games = fake_games()
    with tempfile.TemporaryDirectory() as folder:
        plain, packed = os.path.join(folder, "games.jsonl"), os.path.join(folder, "games.jsonl.gz")
        write(plain, games, packed=False)
        write(packed, games[:2], packed=True)
        write(packed, games[2:], packed=True)           # a later run appends to the same file
        assert load_games(plain) == games and load_games(packed) == games
        assert os.path.getsize(packed) < os.path.getsize(plain)


def test_a_cut_off_last_game_costs_only_that_game():
    games = fake_games()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "games.jsonl.gz")
        write(path, games, packed=True)
        os.truncate(path, os.path.getsize(path) - 40)   # the run was killed in the middle of the last write
        assert load_games(path) == games[:2]


def test_pack_is_lossless_and_refuses_to_overwrite():
    games = fake_games()
    with tempfile.TemporaryDirectory() as folder:
        plain = os.path.join(folder, "games.jsonl")
        write(plain, games, packed=False)
        new, before, after = pack(plain)
        assert new == plain + ".gz" and after < before and load_games(new) == games
        try:
            pack(plain)
        except SystemExit:
            pass
        else:
            raise AssertionError("expected SystemExit")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
