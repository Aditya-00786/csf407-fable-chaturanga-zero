"""Read-only API over training runs, for the web app's Watch training view.

It reads what `chaturanga.training` writes to runs/<name>/ (status.json, metrics.jsonl and the
self-play game files) and never changes anything, so it works the same while a run is going,
after it stopped, or on a run copied from Colab.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from chaturanga.selfplay import read_games

router = APIRouter(prefix="/api/training")

ACTIVE = {"starting", "selfplay", "training", "evaluating", "generation_done"}
# A run updates its status every few seconds; one silent for longer has stopped (closed, crashed).
LIVE_WITHIN_SECONDS = 300
MAX_LOSS_POINTS = 300
_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")


def runs_dir() -> Path:
    default = Path(__file__).resolve().parent.parent / "runs"
    return Path(os.environ.get("CHATURANGA_RUNS", default))


def _run_dir(name: str) -> Path:
    path = runs_dir() / name
    if not _NAME.match(name) or not (path / "config.json").is_file():
        raise HTTPException(404, f"no training run named {name!r}")
    return path


def _status(path: Path) -> dict:
    try:
        status = json.loads((path / "status.json").read_text())
    except (OSError, ValueError):
        status = {"state": "starting"}
    age = None
    if status.get("updated"):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["updated"])).total_seconds()
        except ValueError:
            pass
    status["name"] = path.name
    status["seconds_since_update"] = age
    status["live"] = status.get("state") in ACTIVE and age is not None and age < LIVE_WITHIN_SECONDS
    return status


def _events(path: Path) -> list[dict]:
    out = []
    try:
        with open(path / "metrics.jsonl") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a line being written right now
    except OSError:
        pass
    return out


def _downsample(points: list[dict], keys: tuple, limit: int = MAX_LOSS_POINTS) -> list[dict]:
    """Average consecutive points into at most `limit` buckets, so the chart stays light."""
    if len(points) <= limit:
        return points
    size = len(points) / limit
    out = []
    for b in range(limit):
        chunk = points[int(b * size):int((b + 1) * size)] or [points[-1]]
        avg = {k: sum(p[k] for p in chunk) / len(chunk) for k in keys}
        out.append({"step": chunk[-1]["step"], "generation": chunk[-1]["generation"], **avg})
    return out


@router.get("/runs")
def list_runs():
    root = runs_dir()
    runs = [_status(p) for p in root.iterdir() if (p / "config.json").is_file()] if root.is_dir() else []
    runs.sort(key=lambda r: r.get("updated") or "", reverse=True)
    return {"runs": runs}


@router.get("/runs/{name}")
def run_detail(name: str):
    path = _run_dir(name)
    events = _events(path)
    keys = ("loss", "value_loss", "policy_loss", "policy_entropy")
    train = [{"step": e["step"], "generation": e["generation"], **{k: e.get(k, 0.0) for k in keys}}
             for e in events if e.get("event") == "train"]
    generations: dict[int, dict] = {}
    for e in events:
        g = e.get("generation")
        if g is None or e.get("event") == "train":
            continue
        row = generations.setdefault(g, {"generation": g, "eval": {}})
        kind = e["event"]
        data = {k: v for k, v in e.items() if k not in ("event", "time", "generation")}
        if kind == "selfplay":
            row.update(data)
        elif kind == "held_out":
            row["held_out"] = data
        elif kind == "eval":
            row["eval"][data.pop("opponent")] = data
        elif kind == "generation":
            row["best"] = data.get("best")
            row["train_loss"] = {k: data.get(k) for k in ("loss", "value_loss", "policy_loss")}
            row["steps"] = data.get("steps")
            row["seconds"] = data.get("seconds")
            row["finished"] = e.get("time")
    return {
        "status": _status(path),
        "config": json.loads((path / "config.json").read_text()),
        "train": _downsample(train, keys),
        "generations": [generations[g] for g in sorted(generations)],
    }


def _games_file(path: Path, generation: int) -> Path:
    file = path / "games" / f"gen_{generation:04d}.jsonl.gz"
    if not file.is_file():
        raise HTTPException(404, f"no self-play games for generation {generation}")
    return file


@router.get("/runs/{name}/games/{generation}")
def list_games(name: str, generation: int):
    file = _games_file(_run_dir(name), generation)
    return {"games": [{"index": i, "winner": r.winner, "reason": r.reason, "plies": r.plies, "seed": r.seed}
                      for i, r in enumerate(read_games(str(file)))]}


@router.get("/runs/{name}/games/{generation}/{index}")
def get_game(name: str, generation: int, index: int):
    file = _games_file(_run_dir(name), generation)
    for i, r in enumerate(read_games(str(file))):
        if i == index:
            return {"index": i, "generation": generation, "moves": r.moves, "winner": r.winner,
                    "reason": r.reason, "plies": r.plies, "network": r.generation}
    raise HTTPException(404, f"generation {generation} has no game {index}")
