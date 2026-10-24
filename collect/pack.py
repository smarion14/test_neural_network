"""Gzip a plain games file: about 3.5x smaller, and nothing is lost.

    python -m collect.pack data/games.jsonl        ->  data/games.jsonl.gz

The original is left where it is. Check the new file, then delete the original yourself.
`python -m collect.riot` keeps adding games to a .gz file.
"""
import gzip
import os
import sys


def pack(path):
    """Write path + '.gz' with exactly the same content. Returns (new path, plain bytes, gzipped bytes)."""
    packed_path = path + ".gz"
    if os.path.exists(packed_path):
        raise SystemExit(f"{packed_path} already exists. Move it away first.")
    with open(path, "rb") as f:
        raw = f.read()
    packed = gzip.compress(raw)
    if gzip.decompress(packed) != raw:
        raise SystemExit("The compressed copy does not match the original. Nothing was written.")
    with open(packed_path, "wb") as f:
        f.write(packed)
    return packed_path, len(raw), len(packed)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m collect.pack data/games.jsonl")
    new, before, after = pack(sys.argv[1])
    print(f"{new}: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({before / after:.1f}x smaller)")
