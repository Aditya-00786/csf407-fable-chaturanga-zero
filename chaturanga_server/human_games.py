"""Games people play against the computer in the web app, kept for the Agent vs Human results.

The app sends each game it recorded once it is over: played to the end, resigned, or left before
the end (a new game started, say), which counts as resigning. The engine replays every game before
it is kept and works out the result itself. Games are kept one JSON line each in
results/human_games/<player>.jsonl, one file per player, so teammates committing their own games
never edit the same file. A game is known by the id the app gave it: recorded twice, or imported
from two files, it is kept once.

Teammates playing on their own laptops send their games as a file (Export on the Results page),
checked game by game when it is imported here, or commit their file.
"""

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError

from chaturanga import GameState, parse_move
from chaturanga.agents import AGENTS, LEVELS, MILESTONES

router = APIRouter(prefix="/api/human-games")

ROOT = Path(__file__).resolve().parent.parent
FORMAT = "chaturangzero-human-games"
MAX_MOVES = 1000
COLORS = {"white": 1, "black": -1}
NAMES = {1: "white", -1: "black"}


def games_dir() -> Path:
    return Path(os.environ.get("CHATURANGA_HUMAN_GAMES", ROOT / "results" / "human_games"))


class Network(BaseModel):
    run: Optional[str] = Field(None, max_length=100)
    generation: Optional[int] = None
    fingerprint: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$")


class GameIn(BaseModel):
    """A game as the app recorded it. Its result is not taken from here: the engine works it out."""
    id: str = Field(pattern=r"^[A-Za-z0-9-]{8,64}$")
    player: str = Field(min_length=1, max_length=40)
    human: Literal["white", "black"]
    opponent: str
    level: Optional[str] = None
    network: Optional[Network] = None
    network_changed: bool = False  # the run in Play published a new network during the game
    moves: List[str] = Field(default_factory=list, max_length=MAX_MOVES)
    ended: Literal["over", "resigned", "left"] = "over"
    takebacks: int = Field(0, ge=0, le=MAX_MOVES)
    started: datetime
    finished: datetime


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def check(game: GameIn) -> dict:
    """The game as it is kept, with the engine's result. ValueError says what is wrong with it."""
    player = " ".join(game.player.split())
    if not player or not player.isprintable():
        raise ValueError("the player's name is empty or has control characters")
    if game.opponent not in AGENTS:
        raise ValueError(f"unknown opponent {game.opponent!r}")
    zero = game.opponent == "zero"
    if zero and game.level not in LEVELS:
        raise ValueError(f"unknown level {game.level!r}")
    state = GameState.initial()
    for i, text in enumerate(game.moves):
        if state.is_terminal():
            raise ValueError(f"move {i + 1} ({text}) comes after the game ended")
        try:
            move = parse_move(text)
        except ValueError as exc:
            raise ValueError(f"move {i + 1}: {exc}") from None
        if not state.is_legal(move):
            raise ValueError(f"move {i + 1} ({text}) is illegal")
        state = state.apply(move)
    outcome = state.outcome()
    human = COLORS[game.human]
    if game.ended == "over":
        if outcome is None:
            raise ValueError("the game is not over")
        winner, reason = outcome.winner, outcome.reason
    else:
        if outcome is not None:
            raise ValueError(f"it was {game.ended} after the game was over")
        if (len(game.moves) + (human == 1)) // 2 == 0:
            raise ValueError(f"it was {game.ended} before the player made a move")
        winner, reason = -human, game.ended
    return {
        "id": game.id, "player": player, "human": game.human, "opponent": game.opponent,
        "level": game.level if zero else None,
        "network": game.network.model_dump() if zero and game.network else None,
        "network_changed": zero and game.network_changed,
        "result": "draw" if winner is None else "win" if winner == human else "loss",
        "reason": reason, "winner": None if winner is None else NAMES[winner],
        "plies": len(game.moves), "takebacks": game.takebacks,
        "started": _iso(game.started), "finished": _iso(game.finished), "moves": game.moves,
    }


# ---------------------------------------------------------------------- the files

_lock = threading.Lock()


def slug(player: str) -> str:
    """The player's file name: their name in lower case, letters and digits only."""
    s = re.sub(r"[^a-z0-9]+", "-", player.lower()).strip("-")[:40].strip("-")
    return s or "player-" + hashlib.sha1(player.encode()).hexdigest()[:8]


def _read(path: Path) -> list[dict]:
    games = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            game = json.loads(line)
        except ValueError:
            continue
        if isinstance(game, dict) and "id" in game and "result" in game:
            games.append(game)
    return games


def all_games(folder: Optional[Path] = None) -> list[dict]:
    folder = folder or games_dir()
    return [g for path in sorted(folder.glob("*.jsonl")) for g in _read(path)]


def add(games: list[dict]) -> tuple[int, int]:
    """Keep the games not kept already (by id); (how many were added, how many were kept already).
    A player's file is kept in the order the games finished, so the same games give the same file."""
    with _lock:
        folder = games_dir()
        known = {g["id"] for g in all_games(folder)}
        new: dict[str, list[dict]] = {}
        for g in games:
            if g["id"] not in known:
                known.add(g["id"])
                new.setdefault(slug(g["player"]), []).append(g)
        folder.mkdir(parents=True, exist_ok=True)
        for name, items in new.items():
            path = folder / f"{name}.jsonl"
            kept = (_read(path) if path.exists() else []) + items
            kept.sort(key=lambda g: (g.get("finished") or "", g["id"]))
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in kept), encoding="utf-8")
            tmp.replace(path)
        added = sum(len(items) for items in new.values())
        return added, len(games) - added


# ---------------------------------------------------------------------- results

LEVEL_ORDER = list(LEVELS)


def opponent_label(game: dict) -> str:
    if game["opponent"] != "zero":
        return AGENTS[game["opponent"]].name if game["opponent"] in AGENTS else game["opponent"]
    level = LEVELS[game["level"]].name if game.get("level") in LEVELS else str(game.get("level"))
    net = game.get("network") or {}
    milestone = MILESTONES.get(net.get("fingerprint") or "")
    if milestone:
        which = f"{milestone} agent"
    elif net.get("run"):
        which = f"{net['run']}, generation {net.get('generation')}"
    else:
        which = "an unknown network"
    return f"ChaturangZero {level}, {which}"


def _opponent_key(game: dict) -> tuple:
    net = game.get("network") or {}
    network = (net.get("fingerprint") or f"{net.get('run')}/{net.get('generation')}") if game["opponent"] == "zero" else ""
    return game["opponent"], game.get("level") or "", network


def _tally(games: list[dict]) -> dict:
    t = {"games": len(games), "win": 0, "draw": 0, "loss": 0}
    for g in games:
        t[g["result"]] += 1
    t["score"] = (t["win"] + 0.5 * t["draw"]) / len(games) if games else None
    return t


def _by_player(games: list[dict]) -> list[dict]:
    players: dict[str, list[dict]] = {}
    for g in games:
        players.setdefault(slug(g["player"]), []).append(g)
    rows = [{"player": items[-1]["player"], "file": f"{name}.jsonl", **_tally(items)} for name, items in players.items()]
    return sorted(rows, key=lambda r: (-r["games"], r["player"].lower()))


def summary(games: list[dict]) -> dict:
    """The people's results against each opponent (the trained agent apart for each level and network),
    overall and by player."""
    games = sorted(games, key=lambda g: (g.get("finished") or "", g["id"]))
    groups: dict[tuple, list[dict]] = {}
    for g in games:
        groups.setdefault(_opponent_key(g), []).append(g)
    order = list(AGENTS)

    def rank(item):  # the trained agent first, a milestone's network before others, then by level
        (opponent, level, network), _ = item
        return (opponent != "zero", network not in MILESTONES, network,
                LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 9,
                order.index(opponent) if opponent in order else 9)

    opponents = []
    for (opponent, level, _), items in sorted(groups.items(), key=rank):
        net = items[-1].get("network")
        opponents.append({"opponent": opponent, "level": level or None, "network": net,
                          "milestone": MILESTONES.get((net or {}).get("fingerprint") or ""),
                          "label": opponent_label(items[-1]), **_tally(items), "players": _by_player(items)})
    return {"total": len(games), "opponents": opponents, "players": _by_player(games)}


# ---------------------------------------------------------------------- reading files people send

def parse_file(text: str) -> list:
    """The games in a file: one exported from the Results page, or a player's file of games, one
    per line (results/human_games/<player>.jsonl)."""
    text = text.strip().lstrip("﻿")
    try:
        data = json.loads(text)
    except ValueError:
        items = []
        for i, line in enumerate(text.splitlines()):
            if line.strip():
                try:
                    items.append(json.loads(line))
                except ValueError:
                    raise ValueError(f"line {i + 1} is not a game") from None
        return items
    if isinstance(data, dict) and data.get("format") == FORMAT and isinstance(data.get("games"), list):
        return data["games"]
    if isinstance(data, dict) and "moves" in data:
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("it is not a file of ChaturangZero games")


def check_kept(item) -> dict:
    """A game from a file, checked again: its moves replayed, and its result the engine's."""
    if not isinstance(item, dict):
        raise ValueError("it is not a game")
    reason = item.get("reason")
    fields = {**item, "ended": item.get("ended") or (reason if reason in ("resigned", "left") else "over")}
    try:
        game = check(GameIn.model_validate(fields))
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ValueError(f"{'.'.join(map(str, first['loc']))}: {first['msg']}") from None
    if ("result" in item and item["result"] != game["result"]) or (reason is not None and reason != game["reason"]):
        raise ValueError(f"its result ({item.get('result')}, {reason}) is not what its moves give "
                         f"({game['result']}, {game['reason']})")
    return game


# ---------------------------------------------------------------------- the API

def _folder_shown() -> str:
    folder = games_dir().resolve()
    try:
        return folder.relative_to(ROOT).as_posix()
    except ValueError:
        return str(folder)


@router.get("")
def recorded():
    """Every game kept on this computer, newest first, and the results."""
    games = all_games()
    return {"folder": _folder_shown(), **summary(games),
            "games": sorted(games, key=lambda g: (g.get("finished") or "", g["id"]), reverse=True)}


@router.post("")
def record(game: GameIn):
    try:
        kept = check(game)
    except ValueError as exc:
        raise HTTPException(400, f"The game could not be saved: {exc}.") from exc
    added, _ = add([kept])
    mine = slug(kept["player"])
    return {"game": kept, "added": bool(added), "file": f"{_folder_shown()}/{mine}.jsonl",
            "player_games": sum(slug(g["player"]) == mine for g in all_games())}


@router.get("/export")
def export(player: Optional[str] = None):
    """Every game kept here (or one player's) as one file to send."""
    games = sorted(all_games(), key=lambda g: (g.get("finished") or "", g["id"]))
    if player:
        games = [g for g in games if slug(g["player"]) == slug(player)]
    names = sorted({slug(g["player"]) for g in games})
    now = datetime.now(timezone.utc)
    who = names[0] if len(names) == 1 else "all"
    body = {"format": FORMAT, "version": 1, "exported": _iso(now), "games": games}
    return Response(json.dumps(body, ensure_ascii=False, indent=1), media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="chaturangzero-games-{who}-{now.astimezone():%Y-%m-%d}.json"'})


class ImportFile(BaseModel):
    name: str = Field(max_length=300)
    text: str = Field(max_length=50_000_000)


class ImportRequest(BaseModel):
    files: List[ImportFile] = Field(min_length=1, max_length=100)


@router.post("/import")
def import_games(req: ImportRequest):
    """Add the games in files teammates sent. Each is checked again; one kept already is skipped."""
    games, rejected = [], []
    for f in req.files:
        try:
            items = parse_file(f.text)
        except ValueError as exc:
            rejected.append({"file": f.name, "game": None, "reason": str(exc)})
            continue
        for i, item in enumerate(items):
            try:
                games.append(check_kept(item))
            except ValueError as exc:
                rejected.append({"file": f.name, "game": i + 1, "reason": str(exc)})
    added, already = add(games)
    return {"added": added, "already": already, "rejected": rejected, "total": len(all_games())}
