"""API over training runs, for the web app's Watch training page.

It reads what `chaturanga.training` writes to runs/<name>/ (status.json, metrics.jsonl and the
self-play game files), so it works the same while a run is going, after it stopped, or on a run
copied from Colab. Three actions change things: a pause request (written to the run's control.json,
which the run reads), resuming a run that is not running, and starting a new run (both start
scripts/train.py, detached from the server; one run trains at a time), on this Mac or elsewhere:
Colab, Kaggle, or your own server over SSH (scripts/server_run.py starts it there and copies it
back as it goes). The server listens on 127.0.0.1 by default, so only this machine can use them.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, SecretStr

from chaturanga.agents import trained_checkpoint
from chaturanga.selfplay import read_games
from chaturanga.training import TrainConfig, buffer_for, pending_pause, request_pause

router = APIRouter(prefix="/api/training")

ACTIVE = {"starting", "selfplay", "training", "evaluating", "generation_done", "reporting",
          "uploading", "queued", "cloud_running", "downloading",  # a Kaggle run, followed from here
          "preparing"}  # a run on your server, while the server gets ready
# A run updates its status every few seconds; one silent for longer has stopped (closed, crashed).
LIVE_WITHIN_SECONDS = 300
MAX_LOSS_POINTS = 300
_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")


def runs_dir() -> Path:
    default = Path(__file__).resolve().parent.parent / "runs"
    return Path(os.environ.get("CHATURANGA_RUNS", default))


def drive_runs_dir() -> Optional[Path]:
    """Colab runs, in Google Drive as synced to this Mac (None without Google Drive for desktop)."""
    from chaturanga.cloud import drive_folder

    drive = drive_folder()
    return drive / "runs" if drive else None


def _roots() -> list[Path]:
    return [r for r in (runs_dir(), drive_runs_dir()) if r is not None]


def _all_runs() -> list[Path]:
    seen, out = set(), []
    for root in _roots():
        if root.is_dir():
            for p in root.iterdir():
                if (p / "config.json").is_file() and p.name not in seen:
                    seen.add(p.name)
                    out.append(p)
    return out


def _where(path: Path) -> str:
    """Where a run trains: mac, colab, kaggle or server."""
    try:
        return json.loads((path / "launch.json").read_text()).get("where", "mac")
    except (OSError, ValueError):
        drive = drive_runs_dir()
        return "colab" if drive is not None and path.parent == drive else "mac"


def _run_dir(name: str) -> Path:
    if _NAME.match(name):
        for root in _roots():
            path = root / name
            if (path / "config.json").is_file():
                return path
    raise HTTPException(404, f"no training run named {name!r}")


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
    status["where"] = _where(path)
    status["play"] = _play_choice(path)  # which of its networks Play's Trained opponent uses, if any
    status["seconds_since_update"] = age
    if status["where"] == "server":
        # Followed from here by scripts/server_run.py: live while it follows a run training there.
        # Its status is the server's, so its age says nothing when the connection is down.
        notes = _server_notes(path)
        alive = _process_alive({"pid": notes.get("follower_pid"), "host": notes.get("follower_host")})
        status["live"] = status.get("state") in ACTIVE and bool(alive)
        status["server_connected"] = bool(notes.get("connected")) if status["live"] else None
        status["server_host"] = notes.get("host") or _launch_settings(path).get("host")
        status["server_checked"] = notes.get("checked")
    else:
        status["live"] = status.get("state") in ACTIVE and (
            alive if alive is not None else age is not None and age < LIVE_WITHIN_SECONDS)
    status["pause_requested"] = pending_pause(path) if status["live"] else None
    return status


def _server_notes(path: Path) -> dict:
    try:
        return json.loads((path / "server.json").read_text())
    except (OSError, ValueError):
        return {}


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
    sync_play()
    runs = [_status(p) for p in _all_runs()]
    runs.sort(key=lambda r: r.get("updated") or "", reverse=True)
    return {"runs": runs}


# ---------------------------------------------------------------------- the network in Play
#
# One run gives Play's Trained opponent its network: the run's best (as it improves), its latest
# generation (as it trains), or one generation chosen and kept. The choice is kept in the run's
# launch.json ("play"), and the network copied to checkpoints/best.pt, where Play loads it; a note
# beside it (best.pt.source.json) says which run, generation and file it came from.

PlayChoice = Union[Literal["best", "latest"], int, None]


def _play_choice(path: Path) -> PlayChoice:
    launch = _launch_settings(path)
    if "play" in launch:
        return launch["play"]
    # Runs from before the choice: "publish" meant the best; one started from a terminal publishes
    # its best through train.py, as train.py does by default.
    return "best" if launch.get("publish", True) else None


def _play_source(path: Path, choice: PlayChoice) -> Optional[tuple[Path, Optional[int]]]:
    """The checkpoint file a choice stands for, and its generation (None if unknown)."""
    checkpoints = path / "checkpoints"
    if choice == "best":
        best = checkpoints / "best.pt"
        try:
            generation = json.loads((path / "status.json").read_text()).get("best_generation")
        except (OSError, ValueError):
            generation = None
        return (best, generation) if best.exists() else None
    if choice == "latest":
        generations = sorted(int(p.stem[4:]) for p in checkpoints.glob("gen_*.pt"))
        return (checkpoints / f"gen_{generations[-1]:04d}.pt", generations[-1]) if generations else None
    if isinstance(choice, int):
        file = checkpoints / f"gen_{choice:04d}.pt"
        return (file, choice) if file.exists() else None
    return None


def _play_note() -> Path:
    target = trained_checkpoint()
    return target.with_name(target.name + ".source.json")


def _put_in_play(path: Path, choice: PlayChoice) -> None:
    """Copy the network the run's choice stands for to where Play loads it, unless it is there."""
    source = _play_source(path, choice)
    if source is None:
        return
    file, generation = source
    note = {"run": path.name, "choice": choice, "generation": generation, "file": str(file),
            "modified": file.stat().st_mtime}
    try:
        if json.loads(_play_note().read_text()) == note and trained_checkpoint().exists():
            return
    except (OSError, ValueError):
        pass
    target = trained_checkpoint()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    shutil.copyfile(file, tmp)
    tmp.replace(target)  # Play picks it up at its next move, without a restart
    _play_note().write_text(json.dumps(note))


def sync_play() -> None:
    """Keep Play's network the one its run asks for, as the run trains: a run in the cloud or on the
    server trains in its own folder, so its new networks are copied from there. A run training on
    this Mac with "best" chosen publishes its best itself, the moment it finds one."""
    chosen = [(p, c) for p in _all_runs() if (p / "launch.json").exists() and (c := _play_choice(p)) is not None]
    if not chosen:
        return
    # One run feeds Play; should older settings name several, the one with the newest network wins.
    def newest(item):
        source = _play_source(*item)
        return source[0].stat().st_mtime if source else 0.0

    path, choice = max(chosen, key=newest)
    if choice == "best" and _where(path) == "mac":
        return
    _put_in_play(path, choice)


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
    if _where(path) == "kaggle":
        raise HTTPException(409, "A Kaggle run cannot be paused from here: stop its session on Kaggle.")
    request_pause(path, req.when)
    return _status(path)


class StartRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40, description="Run name: letters, digits, - and _")
    generations: int = Field(20, ge=1, le=1000)
    games_per_generation: int = Field(50, ge=2, le=1000)
    simulations: int = Field(50, ge=2, le=2000)
    eval_games: int = Field(20, ge=0, le=200, description="Test games against each opponent; 0 skips testing")
    publish: bool = Field(True, description="Make this run's best network the app's Trained opponent")
    auto_report: bool = Field(False, description="Make the run report as soon as the run finishes")
    where: Literal["mac", "colab", "kaggle", "server"] = Field(
        "mac", description="Train on this Mac, on Colab, on Kaggle or on your server (over SSH)")
    workers: Optional[int] = Field(None, ge=1, le=1024, description="Self-play processes, on your server")
    gpu_workers: Optional[int] = Field(None, ge=0, le=1024,
                                       description="Of them, how many run the network on the server's GPU")
    blocks: int = Field(TrainConfig.blocks, ge=1, le=40, description="Residual blocks in the network")
    channels: int = Field(TrainConfig.channels, ge=8, le=512, description="Filters per convolution")


_NEW_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
GPU_CONCURRENT = 32


def _launch_settings(path: Path) -> dict:
    """How the app started this run: which of its networks Play uses ("play"), and where it trains.
    Runs started from a terminal have none, and publish their best as train.py does by default."""
    try:
        return json.loads((path / "launch.json").read_text())
    except (OSError, ValueError):
        return {}


def _train_command(path: Path, generations: Optional[int], publish: bool) -> list[str]:
    script = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    cmd = [sys.executable, "-u", str(script), "--run", path.name, "--runs-dir", str(path.parent),
           "--publish", str(trained_checkpoint()) if publish else ""]
    if generations:
        cmd += ["--generations", str(generations)]
    if shutil.which("caffeinate"):  # macOS: keep the machine awake while it trains
        cmd = ["caffeinate", "-is", *cmd]
    return cmd


def _live_run() -> Optional[str]:
    """The run training on this Mac, if any (runs in the cloud use other machines)."""
    for p in _all_runs():
        if _where(p) == "mac" and _status(p)["live"]:
            return p.name
    return None


def _spawn(path: Path, generations: Optional[int], publish: bool, status: dict) -> dict:
    """Start scripts/train.py for the run at `path`, detached, and report it as starting."""
    name = path.name
    log = open(path / "train.log", "a")
    try:
        proc = subprocess.Popen(_train_command(path, generations, publish), stdout=log, stderr=subprocess.STDOUT,
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
    _write_status(path, fresh)
    return _status(path)


def _spawn_kaggle(path: Path, publish: bool, status: dict) -> dict:
    """Start scripts/kaggle_run.py for the run, detached: it uploads, starts, follows and fetches."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "kaggle_run.py"
    cmd = [sys.executable, "-u", str(script), "--run", path.name, "--runs-dir", str(path.parent),
           "--publish", str(trained_checkpoint()) if publish else ""]
    log = open(path / "kaggle_run.log", "a")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(Path(__file__).resolve().parent.parent), start_new_session=True)
    finally:
        log.close()
    threading.Thread(target=proc.wait, name=f"wait-kaggle-{path.name}", daemon=True).start()
    fresh = {k: v for k, v in status.items() if k not in ("name", "seconds_since_update", "live", "pause_requested")}
    fresh.update(state="uploading", where="kaggle", pid=proc.pid, host=socket.gethostname(), paused_in=None,
                 paused_after=None, phase_started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _write_status(path, fresh)
    return _status(path)


def _spawn_server(path: Path, generations: Optional[int], status: dict, follow_only: bool = False) -> dict:
    """Start scripts/server_run.py for the run, detached: it gets the server ready, starts the
    training there (or finds it already training) and copies it back as it goes."""
    # A follower still bringing home the end of the run (it paused a moment ago) finishes first.
    old = _server_notes(path)
    for _ in range(100):
        if not _process_alive({"pid": old.get("follower_pid"), "host": old.get("follower_host")}):
            break
        time.sleep(0.2)
    else:
        raise HTTPException(409, "Still copying the run back from the server: try again in a moment.")
    script = Path(__file__).resolve().parent.parent / "scripts" / "server_run.py"
    cmd = [sys.executable, "-u", str(script), "--run", path.name, "--runs-dir", str(path.parent)]
    if generations:
        cmd += ["--generations", str(generations)]
    if follow_only:
        cmd.append("--follow-only")
    log = open(path / "server_run.log", "a")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(Path(__file__).resolve().parent.parent), start_new_session=True)
    finally:
        log.close()
    threading.Thread(target=proc.wait, name=f"wait-server-{path.name}", daemon=True).start()
    # Live at once: the follower's notes name it, and it takes over from here.
    notes = {**_server_notes(path), "follower_pid": proc.pid, "follower_host": socket.gethostname(),
             "connected": True, "host": _launch_settings(path).get("host"),
             "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ended": None}
    (path / "server.json").write_text(json.dumps(notes, indent=2))
    if not follow_only:
        fresh = {k: v for k, v in status.items() if k not in ("name", "seconds_since_update", "live", "pause_requested",
                                                               "server_connected", "server_host", "server_checked")}
        fresh.update(state="preparing", where="server", setup="Connecting to the server", paused_in=None,
                     paused_after=None, error=None, phase_started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if generations:
            fresh["generations"] = generations
        _write_status(path, fresh)
    return _status(path)


def _server_ready() -> dict:
    """The server's settings, if it is connected now; otherwise why not, as an HTTP error."""
    from chaturanga import remote

    settings = remote.settings()
    if not settings:
        raise HTTPException(409, "Connect to your server first: its address, a folder, and its password.")
    if not remote.connected(settings["host"]):
        raise HTTPException(409, f"Not connected to {settings['host']}: connect with its password first.")
    return settings


def _live_server_run() -> Optional[str]:
    for p in _all_runs():
        if _where(p) == "server" and _status(p)["live"]:
            return p.name
    return None


def _write_status(path: Path, status: dict) -> None:
    tmp = path / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    tmp.replace(path / "status.json")


def _wait_for_colab(path: Path, status: dict) -> dict:
    """Ask the next Colab session to train the run, and show it as waiting for one."""
    from chaturanga.cloud import colab_url, request_colab_run

    request_colab_run(path.parent.parent, path.name)
    fresh = {k: v for k, v in status.items() if k not in ("name", "seconds_since_update", "live", "pause_requested")}
    fresh.update(state="waiting", where="colab", paused_in=None, paused_after=None,
                 updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _write_status(path, fresh)
    return {**_status(path), "colab_url": colab_url()}


@router.post("/runs", status_code=202)
def start_run(req: StartRequest):
    """Start a new run: on this Mac (one at a time), on Colab (through Google Drive), on Kaggle, or
    on your server (one at a time there too)."""
    from chaturanga.cloud import kaggle_credentials

    if not _NEW_NAME.match(req.name):
        raise HTTPException(400, "A run name uses letters, digits, - and _, and starts with a letter or digit.")
    if any(p.name == req.name for p in _all_runs()):
        raise HTTPException(409, f"There is already a run named {req.name!r}.")
    if req.where == "mac":
        root = runs_dir()
        busy = _live_run()
        if busy:
            raise HTTPException(409, f"The run {busy!r} is training. Pause it first: one run at a time keeps each fast.")
    elif req.where == "colab":
        root = drive_runs_dir()
        if root is None:
            raise HTTPException(409, "Colab runs need Google Drive for desktop on this Mac, signed in with the Google "
                                     "account you use for Colab: it carries the run between Colab and this app.")
    elif req.where == "server":
        root = runs_dir()
        server = _server_ready()
        if req.workers or req.gpu_workers is not None:  # kept for the next run too
            from chaturanga import remote

            changes = {"workers": req.workers} if req.workers else {}
            if req.gpu_workers is not None:
                changes["gpu_workers"] = req.gpu_workers
            server = remote.save_settings(**changes)
        busy = _live_server_run()
        if busy:
            raise HTTPException(409, f"The run {busy!r} is training on the server. Pause it first: one run at a time "
                                     "keeps each fast.")
    else:
        root = runs_dir()
        if not kaggle_credentials():
            raise HTTPException(409, "Connect Kaggle first: add your Kaggle username and API token.")
    path = root / req.name
    if path.exists():
        raise HTTPException(409, f"There is already a run named {req.name!r}.")
    # A process that runs its network on the GPU plays 32 games side by side: the GPU works best on
    # large batches (measured on the lab server: 168 moves a second against 66 with 8).
    on_gpu = req.where == "server" and bool(server.get("gpu_workers"))
    config = TrainConfig(generations=req.generations, games_per_generation=req.games_per_generation,
                         simulations=req.simulations, eval_games=req.eval_games, auto_report=req.auto_report,
                         buffer_positions=buffer_for(req.games_per_generation), blocks=req.blocks,
                         channels=req.channels, concurrent=GPU_CONCURRENT if on_gpu else TrainConfig.concurrent)
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config.to_json(), indent=2))
    launch = {"play": "best" if req.publish else None, "where": req.where}
    if req.where == "server":  # the run keeps the server it trains on, even if the app's server changes later
        launch.update(host=server["host"], folder=server["folder_abs"], workers=server["workers"],
                      gpu_workers=min(server.get("gpu_workers") or 0, server["workers"]))
    (path / "launch.json").write_text(json.dumps(launch))
    status = {"run": req.name, "generation": 0, "generations": req.generations}
    if req.where == "colab":
        return _wait_for_colab(path, status)
    if req.where == "server":
        return _spawn_server(path, None, status)
    if req.where == "kaggle":
        return _spawn_kaggle(path, req.publish, status)
    return _spawn(path, None, req.publish, status)


@router.post("/runs/{name}/resume", status_code=202)
def resume_run(name: str, req: ResumeRequest):
    path = _run_dir(name)
    status = _status(path)
    if status["live"]:
        raise HTTPException(409, f"run {name!r} is already running")
    where = _where(path)
    if where == "mac":
        busy = _live_run()
        if busy:
            raise HTTPException(409, f"The run {busy!r} is training. Pause it first: one run at a time keeps each fast.")
    if where == "server":
        _server_connection(path)
        busy = _live_server_run()
        if busy:
            raise HTTPException(409, f"The run {busy!r} is training on the server. Pause it first: one run at a time "
                                     "keeps each fast.")
    config = json.loads((path / "config.json").read_text())
    target = req.generations or config.get("generations", 0)
    done = max((int(p.stem[4:]) for p in (path / "checkpoints").glob("gen_*.pt")), default=0)
    if status.get("state") == "finished" and target <= done:
        raise HTTPException(400, f"run {name!r} has finished its {done} generations; ask for more to train further")
    if req.generations:  # train further: the run reads its target from its config
        config["generations"] = req.generations
        (path / "config.json").write_text(json.dumps(config, indent=2))
    publish = _play_choice(path) == "best"  # the run publishes its own best; other choices are synced
    if where == "colab":
        return _wait_for_colab(path, status)
    if where == "kaggle":
        return _spawn_kaggle(path, publish, status)
    if where == "server":
        return _spawn_server(path, req.generations, status)
    return _spawn(path, req.generations, publish, status)


def _server_connection(path: Path) -> dict:
    """The run's server (from its launch settings), checked to be connected now."""
    from chaturanga import remote

    launch = _launch_settings(path)
    host = launch.get("host")
    if not host or not launch.get("folder"):
        raise HTTPException(400, f"run {path.name!r} does not say which server it trains on")
    if not remote.connected(host):
        raise HTTPException(409, f"Not connected to {host}: connect with its password first.")
    return launch


# ---------------------------------------------------------------------- choosing the network in Play

class PlayRequest(BaseModel):
    network: PlayChoice = Field(description='"best" or "latest" (followed as the run trains), a generation '
                                            '(kept), or null: this run stops giving Play its network')


def _set_launch(path: Path, **changes) -> None:
    launch = _launch_settings(path) or {"where": _where(path)}
    launch.update(changes)
    launch.pop("publish", None)  # replaced by "play"
    (path / "launch.json").write_text(json.dumps(launch))


@router.post("/runs/{name}/play")
def choose_play(name: str, req: PlayRequest):
    """Give Play's Trained opponent one of this run's networks (now, and as the run trains for
    "best" and "latest"), and stop the other runs from replacing it; null stops this run giving Play
    its network (Play keeps the one it has). One run feeds Play at a time."""
    path = _run_dir(name)
    status = _status(path)
    if status["live"] and _where(path) == "mac" and req.network != status["play"]:
        raise HTTPException(409, f"“{name}” is training on this Mac: pause it to change the network Play uses.")
    if req.network is not None:
        if _play_source(path, req.network) is None:
            raise HTTPException(400, f"run {name!r} has no network {req.network!r} yet")
        for other in _all_runs():
            if other != path and _play_choice(other) is not None:
                _set_launch(other, play=None)
    _set_launch(path, play=req.network)
    if req.network is not None:
        _put_in_play(path, req.network)
    return _status(path)


# ---------------------------------------------------------------------- cloud accounts

class KaggleAccount(BaseModel):
    username: str = Field(min_length=1, max_length=100, description="Your Kaggle username")
    token: str = Field(min_length=8, max_length=500, description="An API token from kaggle.com/settings/api")


@router.get("/cloud")
def cloud_status():
    """What cloud training can use here: Google Drive for Colab, and a Kaggle account."""
    from chaturanga.cloud import colab_url, kaggle_credentials, kaggle_tool

    drive = drive_runs_dir()
    creds = kaggle_credentials()
    return {"colab": {"drive": str(drive.parent) if drive else None, "url": colab_url()},
            "kaggle": {"connected": creds is not None, "username": creds["username"] if creds else None,
                       "tool": kaggle_tool() is not None},
            "server": _server_status()}


def _server_status() -> dict:
    from chaturanga import remote

    settings = remote.settings() or {}
    host = settings.get("host")
    return {"host": host, "folder": settings.get("folder") or remote.DEFAULT_FOLDER,
            "folder_path": settings.get("folder_abs"), "workers": settings.get("workers"),
            "gpu_workers": settings.get("gpu_workers") or 0,
            "connected": remote.connected(host) if host else False, "facts": settings.get("facts"),
            "hosts": remote.ssh_config_hosts(), "tools": bool(remote.ssh_tool() and remote.rsync_tool())}


class ServerConnect(BaseModel):
    host: str = Field(min_length=1, max_length=200, description="user@host, an address, or a name from ~/.ssh/config")
    folder: Optional[str] = Field(None, max_length=300, description="Where everything goes on the server")
    password: Optional[SecretStr] = Field(None, max_length=1000, description="Used to connect, and not kept")
    workers: Optional[int] = Field(None, ge=1, le=1024, description="Self-play processes on the server")


class ServerSettings(BaseModel):
    folder: Optional[str] = Field(None, max_length=300)
    workers: Optional[int] = Field(None, ge=1, le=1024)


@router.post("/server/connect")
def connect_server(req: ServerConnect):
    """Open the connection to the server with its password (which is not kept), see what it has,
    and follow any run that was training there while this Mac was away."""
    from chaturanga import remote

    host = req.host.strip()
    previous = remote.settings() or {}
    folder = (req.folder or "").strip() or (previous.get("folder") if previous.get("host") == host else None) \
        or remote.DEFAULT_FOLDER
    if not remote.FOLDER.match(folder):
        raise HTTPException(400, "The folder is a path such as ~/sopProject/chaturangzero, without spaces or quotes.")
    if not remote.ssh_tool() or not remote.rsync_tool():
        raise HTTPException(501, "This Mac needs ssh and rsync to train on a server.")
    try:
        remote.connect(host, req.password.get_secret_value() if req.password else None)
        facts = remote.server_facts(host)
        folder_abs = remote.resolve_folder(folder, facts.get("home") or "~")
    except remote.RemoteError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not facts.get("rsync"):
        raise HTTPException(400, f"{host} has no rsync, which copies the runs back and forth: ask whoever runs it to "
                                 "install it.")
    same = previous.get("host") == host
    workers = req.workers or (previous.get("workers") if same else None) or remote.default_workers(facts.get("cores"))
    remote.save_settings(host=host, folder=folder, folder_abs=folder_abs, workers=workers, facts=facts)
    _follow_server_runs()
    return cloud_status()


def _follow_server_runs() -> None:
    """Runs last seen training on a server that no follower is following (this Mac restarted, say):
    follow them again, or bring home how they ended."""
    from chaturanga import remote

    for path in _all_runs():
        if _where(path) != "server":
            continue
        status = _status(path)
        host = _launch_settings(path).get("host")
        if not status["live"] and status.get("state") in ACTIVE and host and remote.connected(host):
            _spawn_server(path, None, status, follow_only=True)


@router.post("/server/settings")
def server_settings(req: ServerSettings):
    """Change the folder or the cores new runs use (a run keeps the folder it started in)."""
    from chaturanga import remote

    settings = remote.settings()
    if not settings:
        raise HTTPException(409, "Connect to your server first.")
    changes: dict = {}
    if req.folder is not None:
        try:
            changes.update(folder=req.folder.strip(),
                           folder_abs=remote.resolve_folder(req.folder.strip(), (settings.get("facts") or {}).get("home", "~")))
        except remote.RemoteError as exc:
            raise HTTPException(400, str(exc)) from exc
    if req.workers is not None:
        changes["workers"] = req.workers
    remote.save_settings(**changes)
    return cloud_status()


@router.post("/server/disconnect")
def disconnect_server():
    """Close the connection. Training on the server carries on; connect again to follow it."""
    from chaturanga import remote

    settings = remote.settings()
    if settings:
        remote.disconnect(settings["host"])
    return cloud_status()


@router.post("/cloud/kaggle")
def connect_kaggle(account: KaggleAccount):
    """Check the token with Kaggle, then keep it in ~/.kaggle/access_token (readable by you only)."""
    from chaturanga.cloud import KaggleError, kaggle, save_kaggle_credentials

    username, token = account.username.strip(), account.token.strip()
    try:
        kaggle("kernels", "list", "--mine", "--page-size", "1", timeout=60,
               credentials={"username": username, "token": token})
    except KaggleError as exc:
        raise HTTPException(400, f"Kaggle did not accept that token: {exc}") from exc
    save_kaggle_credentials(username, token)
    return cloud_status()


@router.delete("/runs/{name}")
def delete_run(name: str):
    """Delete a run that is not training: its games, networks, metrics and logs. The network it may
    have published for the app (checkpoints/best.pt) is a copy, and stays."""
    path = _run_dir(name)
    roots = [r.resolve() for r in _roots()]
    if path.resolve().parent not in roots or path.is_symlink():
        raise HTTPException(400, f"run {name!r} is not a folder inside the runs folders")
    if _status(path)["live"]:
        raise HTTPException(409, f"The run {name!r} is training. Pause it before deleting it.")
    if _where(path) == "colab":  # a request still waiting for Colab goes too
        (path.parent.parent / "requests" / f"{name}.json").unlink(missing_ok=True)
    if _where(path) == "server":  # and its copy on the server
        from chaturanga import remote

        launch = _launch_settings(path)
        if launch.get("host") and launch.get("folder"):
            if not remote.connected(launch["host"]):
                raise HTTPException(409, f"Connect to {launch['host']} first, so the run's files there are deleted too.")
            try:
                remote.delete_run(launch["host"], launch["folder"], name)
            except remote.RemoteError as exc:
                raise HTTPException(502, f"Could not delete the run on the server: {exc}") from exc
    shutil.rmtree(path)
    return {"deleted": name}


# ---------------------------------------------------------------------- the run report

class StrengthRequest(BaseModel):
    games: int = Field(20, ge=2, le=200, description="Games per network and opponent")


def _strength_status(path: Path) -> Optional[dict]:
    from chaturanga.analysis import read_strength

    data = read_strength(path)
    if data and data.get("state") == "running" and _process_alive(data) is False:
        data["state"] = "stopped"  # the process ended without finishing (closed, crashed)
    return data


@router.get("/runs/{name}/report")
def run_report(name: str):
    """The run's saved report (None until one is made) and its strength check, if one was played.
    Reading never makes a report: that waits for a request, or the run's auto_report."""
    from chaturanga.analysis import read_report

    path = _run_dir(name)
    return {"report": read_report(path), "strength": _strength_status(path)}


@router.post("/runs/{name}/report")
def make_run_report(name: str):
    """Make (or remake) the report: findings, material probe and charts. A few seconds."""
    from chaturanga.analysis import make_report

    path = _run_dir(name)
    if _status(path)["live"]:
        raise HTTPException(409, f"The run {name!r} is training. Pause it or let it finish first.")
    if not list((path / "checkpoints").glob("gen_*.pt")):
        raise HTTPException(400, f"run {name!r} has no networks yet")
    return {"report": make_report(path), "strength": _strength_status(path)}


@router.post("/runs/{name}/report/strength", status_code=202)
def start_strength(name: str, req: StrengthRequest):
    """Play the strength check in the background (scripts/report.py --strength-only)."""
    path = _run_dir(name)
    busy = _live_run()
    if busy:
        raise HTTPException(409, f"The run {busy!r} is training. Pause it first: the check needs the whole machine.")
    current = _strength_status(path)
    if current and current.get("state") == "running":
        raise HTTPException(409, "A strength check is already running for this run.")
    if not list((path / "checkpoints").glob("gen_*.pt")):
        raise HTTPException(400, f"run {name!r} has no networks yet")
    script = Path(__file__).resolve().parent.parent / "scripts" / "report.py"
    cmd = [sys.executable, "-u", str(script), "--run", name, "--runs-dir", str(path.parent), "--strength-only",
           "--games", str(req.games)]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-is", *cmd]
    (path / "report").mkdir(exist_ok=True)
    log = open(path / "report" / "strength.log", "a")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(Path(__file__).resolve().parent.parent), start_new_session=True)
    finally:
        log.close()
    threading.Thread(target=proc.wait, name=f"wait-strength-{name}", daemon=True).start()
    # Report it as running at once; the check overwrites this as soon as it starts.
    from chaturanga.analysis import STRENGTH_FILE

    started = {"state": "running", "games": req.games, "pid": proc.pid, "host": socket.gethostname(),
               "networks": [], "results": {}, "done": 0, "total": 0,
               "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (path / STRENGTH_FILE).write_text(json.dumps(started))
    return _strength_status(path)


@router.get("/runs/{name}/charts.png")
def run_charts(name: str):
    """The run's six-panel chart (scripts/plot_training.py), redrawn when its metrics change."""
    from fastapi.responses import FileResponse

    path = _run_dir(name)
    png, metrics = path / "training.png", path / "metrics.jsonl"
    if not metrics.exists():
        raise HTTPException(404, f"run {name!r} has no metrics yet")
    if not png.exists() or png.stat().st_mtime < metrics.stat().st_mtime:
        script = Path(__file__).resolve().parent.parent / "scripts" / "plot_training.py"
        done = subprocess.run([sys.executable, str(script), "--run", name, "--runs-dir", str(path.parent)],
                              capture_output=True, text=True, timeout=120)
        if done.returncode != 0:
            if "matplotlib" in done.stderr:
                raise HTTPException(501, "Drawing the charts needs matplotlib: pip install -e '.[train]'")
            raise HTTPException(500, "The charts could not be drawn.")
    return FileResponse(png, media_type="image/png", filename=f"{name}-training.png")
