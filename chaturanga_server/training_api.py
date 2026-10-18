"""API over training runs, for the web app's Watch training page.

It reads what `chaturanga.training` writes to runs/<name>/ (status.json, metrics.jsonl and the
self-play game files), so it works the same while a run is going, after it stopped, or on a run
copied from Colab. Two actions change things: a pause request (written to the run's control.json,
which the run reads) and resuming a run that is not running (which starts scripts/train.py for
it, detached from the server). The server listens on 127.0.0.1 by default, so only this machine
can use them.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chaturanga.agents import trained_checkpoint
from chaturanga.selfplay import read_games
from chaturanga.training import pending_pause, request_pause

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


def _process_alive(status: dict) -> Optional[bool]:
    """Whether the run's process is alive, when it runs on this machine (None if we cannot tell)."""
    pid = status.get("pid")
    if not pid or status.get("host") != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, OverflowError):
        return True
    return True


def _status(path: Path) -> dict:
    try:
        status = json.loads((path / "status.json").read_text())
    except (OSError, ValueError):
        # A run that has written its config but not yet its first status is still loading.
        try:
            started = datetime.fromtimestamp((path / "config.json").stat().st_mtime, timezone.utc)
        except OSError:
            started = None
        status = {"state": "starting", "updated": started.isoformat(timespec="seconds") if started else None}
    age = None
    if status.get("updated"):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["updated"])).total_seconds()
        except ValueError:
            pass
    alive = _process_alive(status)
    status["name"] = path.name
    status["seconds_since_update"] = age
    status["live"] = status.get("state") in ACTIVE and (
        alive if alive is not None else age is not None and age < LIVE_WITHIN_SECONDS)
    status["pause_requested"] = pending_pause(path) if status["live"] else None
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
    # Training interrupted by a pause is redone on resume with the same step numbers: the later
    # points replace the earlier ones.
    by_step = {e["step"]: {"step": e["step"], "generation": e["generation"], **{k: e.get(k, 0.0) for k in keys}}
               for e in events if e.get("event") == "train"}
    train = [by_step[k] for k in sorted(by_step)]
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


# ---------------------------------------------------------------------- pausing and resuming

class PauseRequest(BaseModel):
    when: Optional[Literal["now", "after_phase", "after_generation"]] = Field(
        description="When to pause; null withdraws a request")


class ResumeRequest(BaseModel):
    generations: Optional[int] = Field(None, ge=1, le=10_000, description="Train up to this many generations")


@router.post("/runs/{name}/pause")
def pause_run(name: str, req: PauseRequest):
    path = _run_dir(name)
    if not _status(path)["live"]:
        raise HTTPException(409, f"run {name!r} is not running")
    request_pause(path, req.when)
    return _status(path)


def _train_command(name: str, generations: Optional[int]) -> list[str]:
    script = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    cmd = [sys.executable, "-u", str(script), "--run", name, "--runs-dir", str(runs_dir()),
           "--publish", str(trained_checkpoint())]
    if generations:
        cmd += ["--generations", str(generations)]
    if shutil.which("caffeinate"):  # macOS: keep the machine awake while it trains
        cmd = ["caffeinate", "-is", *cmd]
    return cmd


@router.post("/runs/{name}/resume", status_code=202)
def resume_run(name: str, req: ResumeRequest):
    path = _run_dir(name)
    status = _status(path)
    if status["live"]:
        raise HTTPException(409, f"run {name!r} is already running")
    config = json.loads((path / "config.json").read_text())
    target = req.generations or config.get("generations", 0)
    done = max((int(p.stem[4:]) for p in (path / "checkpoints").glob("gen_*.pt")), default=0)
    if status.get("state") == "finished" and target <= done:
        raise HTTPException(400, f"run {name!r} has finished its {done} generations; ask for more to train further")
    log = open(path / "train.log", "a")
    try:
        proc = subprocess.Popen(_train_command(name, req.generations), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=str(Path(__file__).resolve().parent.parent),
                                start_new_session=True)  # survives the API server restarting
    finally:
        log.close()
    # Show "starting" at once; the run overwrites this as soon as it has loaded.
    fresh = {k: v for k, v in status.items() if k not in ("name", "seconds_since_update", "live", "pause_requested")}
    fresh.update(state="starting", pid=proc.pid, host=socket.gethostname(), paused_in=None, paused_after=None,
                 updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if req.generations:
        fresh["generations"] = req.generations
    tmp = path / "status.json.tmp"
    tmp.write_text(json.dumps(fresh))
    tmp.replace(path / "status.json")
    return _status(path)
