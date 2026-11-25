"""Games people play against the computer in the web app, kept for the Agent vs Human results.

The app sends each game it recorded once it is over: played to the end, resigned, or left before
the end (a new game started, say), which counts as resigning. The engine replays every game before
it is kept and works out the result itself. A game keeps only what the results need: the colour
the person played, the opponent (its level and network), the moves, the result and the take-backs.
No names, no times.

Each game is a file of its own, results/human_games/<id>.json, named by the id the app gave it: so
teammates committing their games never edit the same file, and a game recorded twice, or imported
from two files, is kept once. Teammates playing on their own laptops send their games as one file
(Export on the Results page), checked game by game when it is imported here, or commit them.
"""

import json
import os
import threading
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


def games_dir() -> Path:
    return Path(os.environ.get("CHATURANGA_HUMAN_GAMES", ROOT / "results" / "human_games"))


class Network(BaseModel):
    run: Optional[str] = Field(None, max_length=100)
    generation: Optional[int] = None
    fingerprint: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$")


class GameIn(BaseModel):
    """A game as the app recorded it. Its result is not taken from here: the engine works it out."""
    id: str = Field(pattern=r"^[A-Za-z0-9-]{8,64}$")
    human: Literal["white", "black"]
    opponent: str
    level: Optional[str] = None
    network: Optional[Network] = None
    moves: List[str] = Field(default_factory=list, max_length=MAX_MOVES)
    ended: Literal["over", "resigned", "left"] = "over"
    takebacks: int = Field(0, ge=0, le=MAX_MOVES)


def check(game: GameIn) -> dict:
    """The game as it is kept, with the engine's result. ValueError says what is wrong with it."""
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
            raise ValueError(f"it was {game.ended} before the person made a move")
        winner, reason = -human, game.ended
    return {
        "id": game.id, "human": game.human, "opponent": game.opponent, "level": game.level if zero else None,
        "network": game.network.model_dump() if zero and game.network else None,
        "result": "draw" if winner is None else "win" if winner == human else "loss", "reason": reason,
        "takebacks": game.takebacks, "moves": game.moves,
    }


# ---------------------------------------------------------------------- the files

_lock = threading.Lock()


def _read(path: Path) -> Optional[dict]:
    try:
        game = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return game if isinstance(game, dict) and "result" in game and "moves" in game else None


def all_games(folder: Optional[Path] = None) -> list[dict]:
    """Every game kept, the most recently added first (by its file's time on this computer; the
    games themselves keep no times)."""
    found = []
    for path in (folder or games_dir()).glob("*.json"):
        game = _read(path)
        if game is not None:
            found.append((-path.stat().st_mtime, path.stem, game))
    return [game for *_, game in sorted(found, key=lambda f: f[:2])]


def add(games: list[dict]) -> tuple[int, int]:
    """Keep the games not kept already (by id); (how many were added, how many were kept already)."""
    with _lock:
        folder = games_dir()
        folder.mkdir(parents=True, exist_ok=True)
        added = 0
        for g in games:
            path = folder / f"{g['id']}.json"
            if path.exists():
                continue
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(g) + "\n", encoding="utf-8")
            tmp.replace(path)
            added += 1
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


def summary(games: list[dict]) -> dict:
    """The people's results against each opponent: the trained agent apart for each level and network."""
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
        net = items[0].get("network")
        opponents.append({"opponent": opponent, "level": level or None, "network": net,
                          "milestone": MILESTONES.get((net or {}).get("fingerprint") or ""),
                          "label": opponent_label(items[0]), **_tally(items)})
    return {"total": len(games), "opponents": opponents}


# ---------------------------------------------------------------------- reading files people send

def parse_file(text: str) -> list:
    """The games in a file: one exported from the Results page, one game's own file
    (results/human_games/<id>.json), or games one per line."""
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
    """A game from a file, checked again: its moves replayed, and its result the engine's. Anything
    else in it (a name, say, from another version of the app) is left out."""
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
    """Every game kept on this computer, the most recently added first, and the results."""
    games = all_games()
    return {"folder": _folder_shown(), **summary(games), "games": games}


@router.post("")
def record(game: GameIn):
    try:
        kept = check(game)
    except ValueError as exc:
        raise HTTPException(400, f"The game could not be saved: {exc}.") from exc
    added, _ = add([kept])
    return {"game": kept, "added": bool(added), "total": len(all_games())}


@router.get("/export")
def export():
    """Every game kept here, as one file to send."""
    body = {"format": FORMAT, "version": 1, "games": sorted(all_games(), key=lambda g: g["id"])}
    return Response(json.dumps(body, indent=1), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="chaturangzero-games.json"'})


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
