"""Collect ranked solo-queue games from the Riot API into data/games.jsonl.gz.

    set RIOT_API_KEY=RGAPI-...        (PowerShell: $env:RIOT_API_KEY="RGAPI-...")
    python -m collect.riot --per-tier 2500

Resumable: games already in the output file are skipped, and every game is written the
moment it is fetched, so Ctrl-C (or a key that expired mid-run) loses nothing.
A development key allows 100 requests per 2 minutes, so this runs at about 0.4 games/s.
Each game keeps only a compact per-minute summary (see nn/timeline.py): about 16 KB as plain
text, about 5 KB gzipped. An --out path ending in .gz is gzipped (one gzip member per game, so
appending and crashes stay safe); any other path is plain text. nn.load_games reads both.
"""
import argparse
import gzip
import json
import os
import random
import time

from nn.timeline import TIERS, load_games

PLATFORM_TO_REGION = {"na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
                      "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
                      "kr": "asia", "jp1": "asia", "oc1": "sea", "sg2": "sea", "ph2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea"}
MIN_SECONDS_BETWEEN_REQUESTS = 1.25     # 100 requests / 120 s with a little headroom
APEX = {"MASTER"}                        # no divisions; fetched from a different endpoint
DRAGON, HERALD = "DRAGON", "RIFTHERALD"


def compact(match, timeline, tier):
    """Reduce a Match-V5 match + timeline to the record nn.timeline.build_features reads.

    Returns None for games that should not be used (not a full-length game).
    """
    info, frames = match["info"], timeline["info"]["frames"]
    if info["gameDuration"] < 300 or len(frames) < 6:
        return None
    players = sorted(info["participants"], key=lambda p: p["participantId"])
    if [p["teamId"] for p in players] != [100] * 5 + [200] * 5:
        raise ValueError(f"unexpected team layout in {match['metadata']['matchId']}")

    events = []
    for frame in frames:
        for e in frame["events"]:
            t, kind = e["timestamp"] / 60000, e["type"]
            if kind == "CHAMPION_KILL":
                events.append(["kill", t, e["victimId"] - 1, len(e.get("assistingParticipantIds", []))])
            elif kind == "ELITE_MONSTER_KILL" and e.get("monsterType") in (DRAGON, HERALD):
                name = "dragon" if e["monsterType"] == DRAGON else "herald"
                events.append([name, t, 0 if e["killerTeamId"] == 100 else 1])
            elif kind == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                events.append(["tower", t, 0 if e["teamId"] == 200 else 1])   # teamId lost the tower
            elif kind == "WARD_PLACED" and e.get("creatorId"):
                events.append(["ward", t, e["creatorId"] - 1])
            elif kind == "WARD_KILL" and e.get("killerId"):
                events.append(["wardkill", t, e["killerId"] - 1])

    def stats(f, pid):
        s = f["participantFrames"][str(pid)]
        return [s["totalGold"], s["xp"], s["minionsKilled"], s["jungleMinionsKilled"], s["level"]]

    return {"id": match["metadata"]["matchId"], "tier": tier,
            "patch": ".".join(info["gameVersion"].split(".")[:2]),
            "blueWin": int(players[0]["win"]),
            "champs": [p["championId"] for p in players],
            "roles": [p["teamPosition"] for p in players],
            "frames": [[stats(f, pid) for pid in range(1, 11)] for f in frames],
            "events": events}


class Riot:
    def __init__(self, key, platform):
        import requests                       # only needed to collect, not to train or test
        self.http, self.key, self.last = requests, key, 0.0
        self.platform = f"https://{platform}.api.riotgames.com"
        self.region = f"https://{PLATFORM_TO_REGION[platform]}.api.riotgames.com"

    def get(self, url, **params):
        for attempt in range(6):
            wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.http.get(url, params=params, headers={"X-Riot-Token": self.key}, timeout=30)
            except self.http.exceptions.RequestException as e:   # network blip: wait and retry
                print(f"network error, retrying: {type(e).__name__}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise SystemExit(f"Riot rejected the key ({r.status_code}). Dev keys expire every "
                                 "24 hours: make a new one at developer.riotgames.com and rerun.")
            time.sleep(int(r.headers.get("Retry-After", 2 ** attempt)))   # 429 or 5xx: back off
        raise RuntimeError(f"giving up on {url}")

    def players(self, tier):
        """Endless stream of puuids from random pages of a tier's ladder."""
        league = f"{self.platform}/lol/league/v4"
        while True:
            if tier in APEX:
                entries = (self.get(f"{league}/{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5") or {}).get("entries", [])
            else:
                entries = self.get(f"{league}/entries/RANKED_SOLO_5x5/{tier}/{random.choice(['I', 'II', 'III', 'IV'])}",
                                   page=random.randint(1, 5)) or []
            random.shuffle(entries)
            for entry in entries:
                if entry.get("puuid"):
                    yield entry["puuid"]


def write_game(out, game, packed):
    """Append one game to `out` (opened "ab") as a line of JSON. packed: as its own gzip member, so the
    game is complete on disk the moment it is written, and gzip files can be appended to member by member."""
    line = (json.dumps(game, separators=(",", ":")) + "\n").encode()
    out.write(gzip.compress(line) if packed else line)
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=2500)
    ap.add_argument("--platform", default=os.environ.get("RIOT_PLATFORM", "na1"))
    ap.add_argument("--out", default="data/games.jsonl.gz", help="a path ending in .gz is gzipped")
    args = ap.parse_args()
    key = os.environ.get("RIOT_API_KEY") or raise_missing_key()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen, per_tier = set(), {t: 0 for t in TIERS}
    if os.path.exists(args.out):
        for g in load_games(args.out):
            seen.add(g["id"])
            per_tier[g["tier"]] += 1
    print("already collected:", {t: n for t, n in per_tier.items() if n})

    riot = Riot(key, args.platform)
    with open(args.out, "ab") as out:
        for tier in TIERS:
            for puuid in riot.players(tier):
                if per_tier[tier] >= args.per_tier:
                    break
                ids = riot.get(f"{riot.region}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                               queue=420, type="ranked", count=5) or []
                for match_id in ids:
                    if match_id in seen or per_tier[tier] >= args.per_tier:
                        continue
                    seen.add(match_id)
                    match = riot.get(f"{riot.region}/lol/match/v5/matches/{match_id}")
                    timeline = match and riot.get(f"{riot.region}/lol/match/v5/matches/{match_id}/timeline")
                    game = match and timeline and compact(match, timeline, tier)
                    if game:
                        write_game(out, game, packed=args.out.endswith(".gz"))
                        per_tier[tier] += 1
                        if per_tier[tier] % 50 == 0:
                            print(f"{tier}: {per_tier[tier]}/{args.per_tier}")


def raise_missing_key():
    raise SystemExit("Set RIOT_API_KEY first (free development key: developer.riotgames.com).")


if __name__ == "__main__":
    main()
