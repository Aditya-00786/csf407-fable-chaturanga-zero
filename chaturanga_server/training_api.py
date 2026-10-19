"""API over training runs, for the web app's Watch training page.

It reads what `chaturanga.training` writes to runs/<name>/ (status.json, metrics.jsonl and the
self-play game files), so it works the same while a run is going, after it stopped, or on a run
copied from Colab. Three actions change things: a pause request (written to the run's control.json,
which the run reads), resuming a run that is not running, and starting a new run (both start
scripts/train.py, detached from the server; one run trains at a time). The server listens on
127.0.0.1 by default, so only this machine can use them.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chaturanga.agents import trained_checkpoint
from chaturanga.selfplay import read_games
from chaturanga.training import TrainConfig, pending_pause, request_pause

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
    return {"games": [{"index": i, "winner": r.winner, "reason": r.reason, "plies": r.plies, "seed": r.seed,
                       "endgame": bool(r.start_fen)}
                      for i, r in enumerate(read_games(str(file)))]}


@router.get("/runs/{name}/games/{generation}/{index}")
def get_game(name: str, generation: int, index: int):
    file = _games_file(_run_dir(name), generation)
    for i, r in enumerate(read_games(str(file))):
        if i == index:
            return {"index": i, "generation": generation, "moves": r.moves, "winner": r.winner,
                    "reason": r.reason, "plies": r.plies, "network": r.generation, "start_fen": r.start_fen}
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


class StartRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40, description="Run name: letters, digits, - and _")
    generations: int = Field(20, ge=1, le=1000)
    games_per_generation: int = Field(50, ge=2, le=1000)
    simulations: int = Field(50, ge=2, le=2000)
    eval_games: int = Field(20, ge=0, le=200, description="Test games against each opponent; 0 skips testing")
    publish: bool = Field(True, description="Make this run's best network the app's Trained opponent")


_NEW_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


def _launch_settings(path: Path) -> dict:
    """How the app started this run (only whether it publishes, for now). Runs started from a
    terminal have none, and publish as train.py does by default."""
    try:
        return json.loads((path / "launch.json").read_text())
    except (OSError, ValueError):
        return {}


def _train_command(name: str, generations: Optional[int], publish: bool) -> list[str]:
    script = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    cmd = [sys.executable, "-u", str(script), "--run", name, "--runs-dir", str(runs_dir()),
           "--publish", str(trained_checkpoint()) if publish else ""]
    if generations:
        cmd += ["--generations", str(generations)]
    if shutil.which("caffeinate"):  # macOS: keep the machine awake while it trains
        cmd = ["caffeinate", "-is", *cmd]
    return cmd


def _live_run() -> Optional[str]:
    root = runs_dir()
    if not root.is_dir():
        return None
    for p in root.iterdir():
        if (p / "config.json").is_file() and _status(p)["live"]:
            return p.name
    return None


def _spawn(path: Path, generations: Optional[int], publish: bool, status: dict) -> dict:
    """Start scripts/train.py for the run at `path`, detached, and report it as starting."""
    name = path.name
    log = open(path / "train.log", "a")
    try:
        proc = subprocess.Popen(_train_command(name, generations, publish), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=str(Path(__file__).resolve().parent.parent),
                                start_new_session=True)  # survives the API server restarting
    finally:
        log.close()
    # Collect the process when it exits; otherwise it lingers as a zombie, which still counts as
    # alive and would keep a crashed run looking live.
    threading.Thread(target=proc.wait, name=f"wait-{name}", daemon=True).start()
    # Show "starting" at once; the run overwrites this as soon as it has loaded.
    fresh = {k: v for k, v in status.items() if k not in ("name", "seconds_since_update", "live", "pause_requested")}
    fresh.update(state="starting", pid=proc.pid, host=socket.gethostname(), paused_in=None, paused_after=None,
                 updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if generations:
        fresh["generations"] = generations
    tmp = path / "status.json.tmp"
    tmp.write_text(json.dumps(fresh))
    tmp.replace(path / "status.json")
    return _status(path)


@router.post("/runs", status_code=202)
def start_run(req: StartRequest):
    """Start a new run. One run trains at a time: two would each get half the machine."""
    if not _NEW_NAME.match(req.name):
        raise HTTPException(400, "A run name uses letters, digits, - and _, and starts with a letter or digit.")
    path = runs_dir() / req.name
    if path.exists():
        raise HTTPException(409, f"There is already a run named {req.name!r}.")
    busy = _live_run()
    if busy:
        raise HTTPException(409, f"The run {busy!r} is training. Pause it first: one run at a time keeps each fast.")
    config = TrainConfig(generations=req.generations, games_per_generation=req.games_per_generation,
                         simulations=req.simulations, eval_games=req.eval_games)
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config.to_json(), indent=2))
    (path / "launch.json").write_text(json.dumps({"publish": req.publish}))
    return _spawn(path, None, req.publish, {"run": req.name, "generation": 0, "generations": req.generations})


@router.post("/runs/{name}/resume", status_code=202)
def resume_run(name: str, req: ResumeRequest):
    path = _run_dir(name)
    status = _status(path)
    if status["live"]:
        raise HTTPException(409, f"run {name!r} is already running")
    busy = _live_run()
    if busy:
        raise HTTPException(409, f"The run {busy!r} is training. Pause it first: one run at a time keeps each fast.")
    config = json.loads((path / "config.json").read_text())
    target = req.generations or config.get("generations", 0)
    done = max((int(p.stem[4:]) for p in (path / "checkpoints").glob("gen_*.pt")), default=0)
    if status.get("state") == "finished" and target <= done:
        raise HTTPException(400, f"run {name!r} has finished its {done} generations; ask for more to train further")
    return _spawn(path, req.generations, _launch_settings(path).get("publish", True), status)


@router.delete("/runs/{name}")
def delete_run(name: str):
    """Delete a run that is not training: its games, networks, metrics and logs. The network it may
    have published for the app (checkpoints/best.pt) is a copy, and stays."""
    path = _run_dir(name)
    root = runs_dir().resolve()
    if path.resolve().parent != root or path.is_symlink():
        raise HTTPException(400, f"run {name!r} is not a folder inside {root}")
    if _status(path)["live"]:
        raise HTTPException(409, f"The run {name!r} is training. Pause it before deleting it.")
    shutil.rmtree(path)
    return {"deleted": name}
