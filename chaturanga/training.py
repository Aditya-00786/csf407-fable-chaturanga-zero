"""The training loop (Milestone 3): self-play, train, evaluate, repeat.

Each generation g:
  1. self-play: the network of generation g-1 plays `games_per_generation` games against itself;
  2. teammates' games waiting in `incoming/` are checked and added (see `scripts/selfplay.py`);
  3. training: batches from the replay buffer (enough to sample each new position about
     `sample_reuse` times, at most `steps_per_generation`), giving network g;
  4. evaluation: network g plays network g-1 and the Greedy and Random baselines.

Self-play always uses the newest network, as in AlphaZero (no gating). The best network, the one
that scores highest against Greedy and Random together (later generations win ties), is copied
to `best.pt` and, when asked, published where the web app picks it up.

Everything a run produces lives in its directory, so a run stops and resumes at any point:

    runs/<name>/
      config.json            the run's settings
      metrics.jsonl          one JSON object per event: training losses, self-play results, evaluations
      status.json            what the run is doing now (read by the app's Watch training view)
      control.json           a pause request waiting to be honoured (see `request_pause`)
      games/gen_0001.jsonl.gz  self-play games of each generation (teammates' games in games/contrib/);
                             gen_NNNN.partial.jsonl.gz holds the finished games of one still being played
      incoming/              drop teammates' game files here; they are added at the next generation
      checkpoints/gen_0000.pt  one network per generation, best.pt, and train_state.pt (optimiser)

Pausing. `request_pause(run_dir, when)` asks a running run to stop: "now" (as soon as possible:
finished self-play games are kept, games in progress and a half-done training or evaluation are
redone on resume), "after_phase" (once the current self-play, training or evaluation is done) or
"after_generation". Ctrl+C is "now". Starting the run again resumes it from where it paused.
"""

import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .agents import GreedyAgent, RandomAgent
from .config import DEFAULT_RULES
from .constants import WHITE
from .game import GameState, move_to_str
from .mcts import MCTS, MCTSConfig
from .net import NetConfig, NetworkEvaluator, PolicyValueNet, load_checkpoint, new_network, save_checkpoint
from .replay_buffer import ReplayBuffer
from .selfplay import (GameRecord, SelfPlayConfig, append_games, play_games, read_games, read_games_salvaging,
                       summarise, write_games)


@dataclass(frozen=True)
class TrainConfig:
    generations: int = 20             # stop after this many; raise it and resume to train further
    games_per_generation: int = 50
    simulations: int = 50             # MCTS simulations per move in self-play (and the tests, unless set apart)
    # Self-play samples moves in proportion to visits for the first `temperature_plies` plies and at
    # `late_temperature` after. AlphaZero played the most visited move after its opening (0), but at
    # this scale both sides then shuffle into repetition draws (measured: 22 of 28 games with the
    # generation 3 network, against 2 of 28 at 1), and draws teach the value head nothing. At 1,
    # though, loose pieces mostly go unpunished (main-v3's generation 20 took free material 30% of
    # the time), so the value never learns to fear them. 0.5 punished them a third more often (40%)
    # with fewer repetition draws (8% against 17% at 1, 25% at 0; 36 games each).
    temperature_plies: int = 30
    late_temperature: float = 0.5
    # A share of self-play games start from a random position with the Rajas and a few pieces each
    # (see endgames.py): short, mostly decisive games whose uneven material shows what pieces are
    # worth. From the opening alone, the value head of the first run learned nothing in 1,000 games.
    endgame_fraction: float = 0.3
    endgame_max_pieces: int = 5
    # Self-play takes a win in one whenever there is one (see mcts.py); without it, main-v4's
    # generation 60 took one 46% of the time and played on for a median of 12 plies. In 160 games
    # of that network, it decided more games from the opening (75 of 106 against 69) in fewer plies
    # (median 199 against 219). Test games also turn down a move that allows the opponent a win in
    # one, as Play's Hard and Impossible do, but self-play does not: with both sides parrying every
    # mate in one, fewer games were decided (63 of 106) and mates fell from 28 to 8, and the network
    # would no longer see the blunders its search still favours punished.
    take_wins: bool = True
    concurrent: int = 8               # self-play games per process, sharing batched evaluations
    buffer_positions: int = 60_000    # about the last six generations of 50 games; see buffer_for
    batch_size: int = 256
    steps_per_generation: int = 2_000  # at most; see sample_reuse
    # Training steps per generation are sized so the new positions are sampled about this many
    # times on average (a replay ratio). More passes over few games make the value head memorise
    # which game a position came from instead of learning to judge it. 0 always runs the maximum.
    sample_reuse: float = 4.0
    # The value target: the game's result blended with the search's estimate at each position
    # (0: the result alone, as in AlphaZero; 0.5: their average). With the result alone, the value
    # head of the first two runs learned the few hundred games in its buffer by heart (3x below the
    # predict-a-draw loss on them, above it on new games). See replay_buffer.py.
    value_mix: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eval_games: int = 20              # per opponent after each generation; 0 skips evaluation
    # Simulations per move in the test games and the strength check; None: as in self-play. Kept apart
    # so that a deeper self-play search leaves the tests, and their scores, comparable with earlier.
    eval_simulations: Optional[int] = None
    eval_opponents: tuple = ("previous", "greedy", "random")
    # Test games sample their first moves in proportion to visits so they differ; with only 8 such
    # plies, games between successive generations nearly all ended in repetition.
    eval_opening_plies: int = 16
    # Make the run report (findings, material probe, charts and a strength check of `report_games`
    # games per match) as soon as the last generation is done. Off: the report waits for a request.
    auto_report: bool = False
    report_games: int = 20
    blocks: int = NetConfig.blocks
    channels: int = NetConfig.channels
    seed: int = 0

    @property
    def test_simulations(self) -> int:
        return self.eval_simulations or self.simulations

    @property
    def net_config(self) -> NetConfig:
        return NetConfig(blocks=self.blocks, channels=self.channels)

    @property
    def selfplay_config(self) -> SelfPlayConfig:
        return SelfPlayConfig(simulations=self.simulations, temperature_plies=self.temperature_plies,
                              concurrent=self.concurrent, late_temperature=self.late_temperature,
                              endgame_fraction=self.endgame_fraction, endgame_max_pieces=self.endgame_max_pieces,
                              take_wins=self.take_wins)

    def to_json(self) -> dict:
        d = asdict(self)
        d["eval_opponents"] = list(self.eval_opponents)
        return d

    @classmethod
    def from_json(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        if "eval_opponents" in data:
            data["eval_opponents"] = tuple(data["eval_opponents"])
        return cls(**data)


def buffer_for(games_per_generation: int) -> int:
    """A replay buffer for this many games per generation: the default 60,000 positions, or about
    the last three generations when they are larger (a game is about 200 positions). Far more
    games than the network can learn by heart is what makes its value judge positions."""
    return max(TrainConfig.buffer_positions, games_per_generation * 600)


def training_device() -> torch.device:
    """Training sends large batches, where a GPU (CUDA on Colab, MPS on a Mac) pays off."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- pausing

PAUSE_WHEN = ("now", "after_phase", "after_generation")
# Which pause requests take effect at each kind of checkpoint in the loop.
_HONOURED = {"now": {"now"}, "phase": {"now", "after_phase"},
             "generation": {"now", "after_phase", "after_generation"}}


class Paused(Exception):
    """A pause request took effect."""


def request_pause(run_dir, when: Optional[str]) -> None:
    """Ask the run in `run_dir` to pause ("now", "after_phase" or "after_generation"); None
    withdraws the request. The run reads the request from control.json while it works."""
    path = Path(run_dir) / "control.json"
    if when is None:
        path.unlink(missing_ok=True)
        return
    if when not in PAUSE_WHEN:
        raise ValueError(f"when must be one of {PAUSE_WHEN}, got {when!r}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pause": when, "requested": _now()}))
    tmp.replace(path)


def pending_pause(run_dir) -> Optional[str]:
    try:
        when = json.loads((Path(run_dir) / "control.json").read_text()).get("pause")
    except (OSError, ValueError, AttributeError):
        return None
    return when if when in PAUSE_WHEN else None


# ---------------------------------------------------------------------- workers

# The networks loaded in a worker process, by (checkpoint, modification time).
_worker_cache: dict = {}
# Where a worker process evaluates positions. Its search always runs on its own CPU core; the
# network runs there too, or on the GPU for the first `gpu_workers` processes of a run (see
# TrainingRun), so both work at once.
_worker_device = torch.device("cpu")


# Live progress: moves played and games finished, counted by whoever plays them (a worker process
# or the run itself) so the run can report progress between finished batches. None: not counted.
_progress: Optional[tuple] = None


class _Counter:
    """A plain counter with the interface of multiprocessing.Value, for play in one process."""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


def _count(moves: int = 0, games: int = 0) -> None:
    if _progress is None:
        return
    for counter, n in zip(_progress, (moves, games)):
        if n:
            with counter.get_lock():
                counter.value += n


def _init_worker(moves=None, games=None, devices=None, started=None) -> None:
    global _progress, _worker_device
    torch.set_num_threads(1)  # one process per core is faster than threads for batch sizes this small
    # Ctrl+C is for the main process, which pauses the run and stops the workers itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _progress = (moves, games) if moves is not None else None
    if devices and started is not None:  # each process takes the next device as it starts
        with started.get_lock():
            index = started.value
            started.value += 1
        _worker_device = torch.device(devices[index % len(devices)])


def _load_evaluator(checkpoint: str) -> NetworkEvaluator:
    net, _ = load_checkpoint(checkpoint)
    return NetworkEvaluator(net, _worker_device)


def _evaluator_for(checkpoint: str) -> NetworkEvaluator:
    """The network in a checkpoint, loaded once per process (and again when the file changes); up to
    a dozen are kept, as test games, the strength check and the rating ladder play several at once."""
    key = (checkpoint, os.path.getmtime(checkpoint))
    if key not in _worker_cache:
        while len(_worker_cache) >= 12:
            _worker_cache.pop(next(iter(_worker_cache)))
        _worker_cache[key] = _load_evaluator(checkpoint)
    return _worker_cache[key]


def selfplay_task(checkpoint: str, seeds: list[int], config: SelfPlayConfig,
                  generation: int, contributor: str, save_prefix: Optional[str] = None) -> list[dict]:
    """Play the games with these seeds. With `save_prefix`, each game is also saved the moment it
    ends (to <prefix>.<pid>.jsonl.gz), so a pause or crash loses only games still in progress."""
    def on_game(r: GameRecord) -> None:
        if save_prefix:
            append_games(f"{save_prefix}.{os.getpid()}.jsonl.gz", [r])
        _count(games=1)

    records = play_games(_evaluator_for(checkpoint), len(seeds), config, generation=generation,
                         contributor=contributor, on_game=on_game, seeds=seeds,
                         on_move=lambda n: _count(moves=n))
    return [r.to_json() for r in records]


def chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


class SearchPlayer:
    """MCTS with no noise, following the search's rules as Play's Hard level does (it takes a win in
    one, and does not allow one that can be avoided). It samples its first `opening_plies` moves in
    proportion to visits, so that evaluation games between deterministic players still differ."""

    def __init__(self, evaluator, simulations: int, seed: int, opening_plies: int = 8):
        self.mcts = MCTS(evaluator, MCTSConfig(simulations=simulations, take_wins=True, avoid_losses=True), seed=seed)
        self.opening_plies = opening_plies

    def select_move(self, state: GameState) -> int:
        root = self.mcts.search(state)
        return self.mcts.choose_move(root, 1.0 if state.ply < self.opening_plies else 0.0)


def _player(spec: str, seed: int, simulations: int, opening_plies: int):
    if spec == "greedy":
        return GreedyAgent(seed)
    if spec == "random":
        return RandomAgent(seed)
    return SearchPlayer(_evaluator_for(spec), simulations, seed, opening_plies)  # a checkpoint path


def play_matches(games: list, simulations: int, opening_plies: int = 8, keep_moves: bool = False) -> list[dict]:
    """Test games, each (checkpoint, opponent, index, seed) or (checkpoint, opponent, index, seed,
    start): the checkpoint (or "greedy" or "random") plays the opponent ("greedy", "random" or another
    checkpoint), searching with seed + index (the opponent with seed + 7919 * (index + 1)). From the
    opening it plays White in even-numbered games; from a start position (a FEN) it plays the side to
    move there, and both sides play their best moves from the first. Returns one row per game, in order.

    The games are played side by side, whatever their networks and opponents. At each step of the
    searches under way, the positions waiting for the same network, from every game, are evaluated
    in one batch, which a GPU does far faster than one at a time. Each game is played exactly as if
    alone: the same searches, simulation by simulation, the same seeds and the same sampled opening
    moves. (Replaying the tests of main-v4's generation 40 this way gave the same 120 games.)"""
    table = []
    for checkpoint, opponent, i, seed, *rest in games:
        start = rest[0] if rest else None
        state = GameState.from_fen(start) if start else GameState.initial()
        sampled = 0 if start else opening_plies
        table.append({"checkpoint": checkpoint, "opponent": opponent, "index": i, "state": state, "moves": [],
                      "colour": state.side if start else WHITE if i % 2 == 0 else -WHITE,
                      "player": _player(checkpoint, seed + i, simulations, sampled),
                      "other": _player(opponent, seed + 7919 * (i + 1), simulations, sampled)})

    def play(game: dict, move: int) -> None:
        game["state"] = game["state"].apply(move)
        if keep_moves:
            game["moves"].append(move_to_str(move))
        _count(moves=1)

    active = list(table)
    while active:
        searching = []
        for g in active:
            mover = g["player"] if g["state"].side == g["colour"] else g["other"]
            if isinstance(mover, SearchPlayer):
                searching.append((g, mover, mover.mcts.prepare(g["state"])))
            else:
                play(g, mover.select_move(g["state"]))  # Greedy and Random need no search
        for _ in range(simulations):
            waiting: dict = {}
            for _, searcher, root in searching:
                path = searcher.mcts.descend(root)
                if path is not None:  # (None: a finished game, scored exactly)
                    waiting.setdefault(id(searcher.mcts.evaluator), (searcher.mcts.evaluator, []))[1].append((searcher, path))
            for evaluator, items in waiting.values():
                for (searcher, path), (priors, value) in zip(items, evaluator.evaluate_many([p[-1].state for _, p in items])):
                    searcher.mcts.expand_and_backup(path, priors, value)
        for g, searcher, root in searching:
            play(g, searcher.mcts.choose_move(root, 1.0 if g["state"].ply < searcher.opening_plies else 0.0))
        for g in active:
            if g["state"].is_terminal():
                _count(games=1)
        active = [g for g in active if not g["state"].is_terminal()]

    out = []
    for g in table:
        result = g["state"].outcome()
        row = {"checkpoint": g["checkpoint"], "opponent": g["opponent"], "index": g["index"],
               "score": (result.value_for(g["colour"]) + 1) / 2, "reason": result.reason, "plies": g["state"].ply,
               "colour": g["colour"]}
        if keep_moves:
            row["moves"] = g["moves"]
        out.append(row)
    return out


def match_task(checkpoint: str, opponent: str, game_indices: list[int], simulations: int, seed: int,
               opening_plies: int = 8, keep_moves: bool = False) -> list[dict]:
    """Games between a checkpoint and one opponent, side by side (see play_matches)."""
    return play_matches([(checkpoint, opponent, i, seed) for i in game_indices], simulations, opening_plies, keep_moves)


def match_plan(games: list, workers: int, on_gpu: bool) -> list[list]:
    """Test games dealt into tasks, each played side by side. On a GPU, larger batches over fewer
    processes are faster: on the lab server, 120 games over 5 processes (24 each) took 5.8 minutes,
    over 10 processes 8.3, and over 10 processes one opponent at a time (4 each) 17.6. On CPUs, one
    task per worker keeps every core busy. In this process (no workers), tasks of at most 24 games,
    so a pause is not kept waiting long. Dealt round the tasks, so each mixes the opponents."""
    if workers <= 0:
        tasks = max(1, math.ceil(len(games) / 24))
    else:
        tasks = max(1, workers // 2 if on_gpu else workers)
    tasks = min(tasks, max(1, len(games)))
    return [games[k::tasks] for k in range(tasks)]


# ---------------------------------------------------------------------- training steps

def train_steps(net: PolicyValueNet, optimizer: torch.optim.Optimizer, buffer: ReplayBuffer, steps: int,
                batch_size: int, device: torch.device, rng: np.random.Generator,
                on_progress: Optional[Callable[[int, dict], None]] = None, report_every: int = 10,
                before_step: Optional[Callable[[], None]] = None) -> dict:
    """Run `steps` optimiser steps on batches from the buffer. The loss is AlphaZero's: the value's
    squared error plus the policy's cross-entropy against the search's visit shares (weight decay
    is in the optimiser). Returns the mean of each loss over the steps."""
    net.train()
    totals = {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0, "policy_entropy": 0.0}
    window = dict.fromkeys(totals, 0.0)
    for step in range(1, steps + 1):
        if before_step:
            before_step()  # may raise to abandon the training (a pause)
        x, pi, z = buffer.sample(batch_size, rng)
        x, pi, z = (torch.from_numpy(a).to(device) for a in (x, pi, z))
        logits, v = net(x)
        value_loss = F.mse_loss(v, z)
        policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean()
        loss = value_loss + policy_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # The targets' own entropy is the lowest the policy loss can go; the gap is what is left to learn.
        entropy = -(pi * torch.log(pi.clamp_min(1e-12))).sum(1).mean()
        for k, val in (("loss", loss), ("value_loss", value_loss), ("policy_loss", policy_loss), ("policy_entropy", entropy)):
            val = float(val.detach())
            totals[k] += val
            window[k] += val
        if on_progress and (step % report_every == 0 or step == steps):
            n = report_every if step % report_every == 0 else step % report_every
            on_progress(step, {k: v / n for k, v in window.items()})
            window = dict.fromkeys(totals, 0.0)
    net.eval()
    return {k: v / max(1, steps) for k, v in totals.items()}


@torch.inference_mode()
def held_out_loss(net: PolicyValueNet, records: list[GameRecord], device: torch.device,
                  positions: int = 4096, seed: int = 0) -> dict:
    """The losses on games the network has not trained on (a generation's new self-play games,
    before training on them): the honest measure of how well it generalises."""
    fresh = ReplayBuffer(capacity=10**9)
    for r in records:
        fresh.add(r)
    if not len(fresh):
        return {}
    x, pi, z = fresh.sample(min(positions, len(fresh)), np.random.default_rng(seed))
    net.eval()
    logits, v = [], []
    for start in range(0, len(z), 1024):
        lg, vv = net(torch.from_numpy(x[start:start + 1024]).to(device))
        logits.append(lg.float().cpu())
        v.append(vv.float().cpu())
    logits, v = torch.cat(logits), torch.cat(v)
    pi, z = torch.from_numpy(pi), torch.from_numpy(z)
    value_loss = float(F.mse_loss(v, z))
    policy_loss = float(-(pi * F.log_softmax(logits, dim=1)).sum(1).mean())
    # What predicting a draw everywhere would score: the value loss to beat.
    return {"value_loss": value_loss, "policy_loss": policy_loss, "loss": value_loss + policy_loss,
            "value_baseline": float((z ** 2).mean())}


# ---------------------------------------------------------------------- the run

class TrainingRun:
    def __init__(self, directory: str, config: Optional[TrainConfig] = None, workers: int = 0,
                 device: Optional[torch.device] = None, selfplay_device: Optional[torch.device] = None,
                 publish: Optional[str] = None, log: Callable[[str], None] = print, gpu_workers: int = 0):
        self.dir = Path(directory)
        self.name = self.dir.name
        self.workers = workers
        # Of the worker processes, how many evaluate positions on the GPU (the rest on their CPU core).
        self.gpu_workers = min(gpu_workers, workers) if torch.cuda.is_available() else 0
        self.device = device or training_device()
        self.selfplay_device = selfplay_device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.publish = publish
        self.log = log
        for sub in ("games/contrib", "games/rejected", "incoming", "checkpoints"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

        saved = self.dir / "config.json"
        if saved.exists():
            old = TrainConfig.from_json(json.loads(saved.read_text()))
            if config is not None and (config.blocks, config.channels) != (old.blocks, old.channels):
                raise ValueError("the network size of an existing run cannot change")
            config = config or old
        self.config = config or TrainConfig()
        saved.write_text(json.dumps(self.config.to_json(), indent=2))

        self.status: dict = {"run": self.name, "state": "starting", "generation": 0,
                             "generations": self.config.generations, "pid": os.getpid(),
                             "host": socket.gethostname()}
        self.buffer = ReplayBuffer(self.config.buffer_positions, value_mix=self.config.value_mix)
        self.net: Optional[PolicyValueNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.step = 0
        self.best = 0
        self.best_score = -1.0
        self._pending_eval: Optional[dict] = None  # a trained generation whose evaluation was interrupted
        self._paused_after: Optional[str] = None
        self._pool = None
        # Live progress between finished batches: counters the players add to, and the phase's
        # reporter that a ticker thread calls twice a second.
        ctx = mp.get_context("spawn")
        self._moves, self._games = ctx.Value("q", 0), ctx.Value("q", 0)
        self._live: Optional[Callable[[], None]] = None
        self._status_lock = threading.Lock()
        self._plies_per_game = 240.0  # a first guess (an untrained network's games); then measured each generation
        self._eval_plies = {"previous": 280.0, "random": 260.0, "greedy": 130.0}

    # -- paths and records

    def checkpoint(self, generation: int) -> Path:
        return self.dir / "checkpoints" / f"gen_{generation:04d}.pt"

    def games_file(self, generation: int) -> Path:
        return self.dir / "games" / f"gen_{generation:04d}.jsonl.gz"

    def partial_prefix(self, generation: int) -> Path:
        """Finished games of a generation still being played: <prefix>.<pid>.jsonl.gz, one file per process."""
        return self.dir / "games" / f"gen_{generation:04d}.partial"

    def record(self, event: str, **data) -> None:
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"event": event, "time": _now(), **data}) + "\n")

    def set_status(self, **changes) -> None:
        with self._status_lock:  # the ticker thread writes too
            self.status.update(changes, updated=_now())
            tmp = self.dir / "status.json.tmp"
            tmp.write_text(json.dumps(self.status))
            tmp.replace(self.dir / "status.json")

    def _phase(self, expected_moves: Optional[float] = None, games_before: int = 0) -> None:
        """Start a phase's live progress: from now, moves played and games finished (on top of
        `games_before`) are reported twice a second, with the time the phase began."""
        base_moves, base_games = self._moves.value, self._games.value
        expected = round(expected_moves) if expected_moves else None
        self.set_status(phase_started=_now(), moves=0 if expected else None, moves_expected=expected)
        if not expected:
            self._live = None
            return

        def live() -> None:
            changes = {"moves": self._moves.value - base_moves}
            total = self.status.get("games_total")
            if total:
                games = games_before + self._games.value - base_games
                changes["games_done"] = min(total, max(self.status.get("games_done") or 0, games))
            self.set_status(**changes)

        self._live = live

    # -- pausing

    def _pause_due(self, checkpoint: str) -> Optional[str]:
        """The pending request if it takes effect at this kind of checkpoint ("now": anywhere;
        "phase": between phases; "generation": between generations)."""
        when = pending_pause(self.dir)
        return when if when in _HONOURED[checkpoint] else None

    def _check_pause(self, checkpoint: str, after: Optional[str] = None) -> None:
        """Raise Paused if a request takes effect here. `after` names the phase just completed."""
        if self._pause_due(checkpoint):
            self._paused_after = after
            raise Paused()

    # -- setup and resume

    def _completed_generations(self) -> int:
        done = [int(p.stem[4:]) for p in (self.dir / "checkpoints").glob("gen_*.pt")]
        return max(done) if done else -1

    def _load_or_create(self) -> int:
        """Load the newest network (or create generation 0) and refill the buffer. Returns the
        newest trained generation."""
        last = self._completed_generations()
        if last < 0:
            self.net = new_network(self.config.net_config, seed=self.config.seed)
            save_checkpoint(str(self.checkpoint(0)), self.net, generation=0, run=self.name)
            last = 0
        else:
            self.net, _ = load_checkpoint(str(self.checkpoint(last)))
        self.net.to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.config.learning_rate,
                                           weight_decay=self.config.weight_decay)
        state_file = self.dir / "checkpoints" / "train_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu", weights_only=True)
            if state.get("generation") == last:
                self.optimizer.load_state_dict(state["optimizer"])
                # The saved state carries the rate it was saved with: the settings decide, so a run
                # resumed with a new learning rate (or weight decay) trains with it.
                for group in self.optimizer.param_groups:
                    group.update(lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
                self.step = state["step"]
                if not state.get("evaluated", True):
                    self._pending_eval = {"steps": state.get("steps"), "losses": state.get("losses", {})}
            self.best, self.best_score = state.get("best", 0), state.get("best_score", -1.0)

        # Refill the buffer from the newest games backwards, then add them oldest first.
        files = sorted((self.dir / "games").glob("gen_????.jsonl.gz"), reverse=True)
        files += sorted((self.dir / "games" / "contrib").glob("*.jsonl.gz"), key=os.path.getmtime, reverse=True)
        chosen, positions = [], 0
        for path in files:
            if path.parent.name == "games" and int(path.name[4:8]) > last:
                continue  # self-play for a generation that never finished training: added when it resumes
            records = list(read_games(str(path)))
            chosen.append(records)
            positions += sum(r.plies for r in records)
            if positions >= self.config.buffer_positions:
                break
        for records in reversed(chosen):
            for r in records:
                self.buffer.add(r)
        return last

    # -- the worker processes

    def _pool_or_none(self):
        if self.workers <= 0:
            return None
        if self._pool is None:
            # multiprocessing's Pool (unlike concurrent.futures) can stop its workers at once,
            # which "pause now" needs: games in progress are abandoned, not waited for.
            ctx = mp.get_context("spawn")
            devices = ["cuda"] * self.gpu_workers + ["cpu"] * (self.workers - self.gpu_workers)
            self._pool = ctx.Pool(self.workers, initializer=_init_worker,
                                  initargs=(self._moves, self._games, devices, ctx.Value("i", 0)))
        return self._pool

    def _gather(self, pending: list, on_result: Callable) -> None:
        """Wait for pool tasks, handing each result over as it arrives and pausing if asked to."""
        while pending:
            self._check_pause("now")
            still = []
            for task in pending:
                if task.ready():
                    on_result(task.get())
                else:
                    still.append(task)
            pending = still
            if pending:
                time.sleep(0.2)

    def _stop_pool(self, abandon: bool) -> None:
        if self._pool is None:
            return
        if abandon:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()
        self._pool = None

    # -- the phases of a generation

    def _selfplay(self, generation: int) -> list[GameRecord]:
        path = self.games_file(generation)
        if path.exists():  # played before an interruption
            return list(read_games(str(path)))
        n = self.config.games_per_generation
        base = self.config.seed * 1_000_003 + generation * 10_000
        prefix = self.partial_prefix(generation)
        partials = sorted(prefix.parent.glob(prefix.name + ".*.jsonl.gz"))
        # Games finished before a pause or crash are kept; only the missing seeds are played.
        found: dict[int, GameRecord] = {}
        for f in partials:
            for r in read_games_salvaging(str(f)):
                if r.generation == generation - 1 and r.seed is not None:
                    found[r.seed] = r
        records = list(found.values())
        missing = [base + i for i in range(n) if base + i not in found]
        if records:
            self.log(f"  resuming self-play: {len(records)} of {n} games were already played")
        self.set_status(state="selfplay", generation=generation, games_done=len(records), games_total=n)
        self._phase(len(missing) * self._plies_per_game, games_before=len(records))

        def keep(new: list[GameRecord]) -> None:
            records.extend(new)
            self.set_status(games_done=len(records))

        pool = self._pool_or_none()
        if missing and pool is None:
            evaluator = NetworkEvaluator(self.net, self.selfplay_device)
            own = f"{prefix}.{os.getpid()}.jsonl.gz"

            def on_game(r: GameRecord) -> None:
                append_games(own, [r])
                _count(games=1)
                keep([r])

            try:
                play_games(evaluator, len(missing), self.config.selfplay_config, generation=generation - 1,
                           on_game=on_game, seeds=missing, should_stop=lambda: bool(self._pause_due("now")),
                           on_move=lambda k: _count(moves=k))
            finally:
                self.net.to(self.device)
            self._check_pause("now")
        elif missing:
            size = max(1, min(self.config.concurrent, math.ceil(len(missing) / (2 * self.workers))))
            net_path = str(self.checkpoint(generation - 1))
            tasks = [pool.apply_async(selfplay_task, (net_path, seeds, self.config.selfplay_config, generation - 1, "",
                                                      str(prefix)))
                     for seeds in chunks(missing, size)]
            self._gather(tasks, lambda result: keep([GameRecord.from_json(d) for d in result]))
        self._live = None
        if records:
            self._plies_per_game = sum(r.plies for r in records) / len(records)
        records.sort(key=lambda r: r.seed)
        tmp = path.with_suffix(".tmp")
        write_games(str(tmp), records)
        tmp.replace(path)
        for f in prefix.parent.glob(prefix.name + ".*.jsonl.gz"):
            f.unlink()
        return records

    def _take_incoming(self) -> dict:
        """Add teammates' games from incoming/. Games for other rules or with illegal moves are
        rejected. Accepted games are kept in games/contrib/ (and reloaded on resume), rejected
        ones in games/rejected/ for a look."""
        added: dict = {}
        rejected = positions = 0
        for path in sorted((self.dir / "incoming").glob("*.jsonl*")):
            try:
                records = list(read_games(str(path)))
            except (OSError, EOFError, ValueError, TypeError) as exc:
                self.log(f"  skipping {path.name}: {exc}")
                rejected += 1
                path.replace(self.dir / "games" / "rejected" / path.name)
                continue
            good, bad = [], []
            for r in records:
                try:
                    if r.rules != asdict(DEFAULT_RULES):
                        raise ValueError("different rules")
                    positions += self.buffer.add(r)
                    good.append(r)
                    name = r.contributor or "unknown"
                    added[name] = added.get(name, 0) + 1
                except (ValueError, TypeError, KeyError):
                    bad.append(r)
            stem = path.name.split(".jsonl")[0]
            for kept, folder in ((good, "contrib"), (bad, "rejected")):
                if kept:
                    write_games(str(self.dir / "games" / folder / f"{stem}.jsonl.gz"), kept)
            rejected += len(bad)
            path.unlink()
        if added or rejected:
            self.log(f"  teammates' games: {added or 'none'} added, {rejected} rejected")
        return {"added": added, "rejected": rejected, "positions": positions}

    def steps_for(self, new_positions: int) -> int:
        cfg = self.config
        if cfg.sample_reuse <= 0:
            return cfg.steps_per_generation
        return max(1, min(cfg.steps_per_generation, math.ceil(cfg.sample_reuse * new_positions / cfg.batch_size)))

    def _train(self, generation: int, new_positions: int) -> dict:
        steps = self.steps_for(new_positions)
        self.set_status(state="training", step=0, steps_total=steps)
        self._phase()
        rng = np.random.default_rng(self.config.seed * 7_919 + generation)

        def on_progress(step, losses):
            self.record("train", generation=generation, step=self.step + step, **{k: round(v, 5) for k, v in losses.items()})
            self.set_status(step=step)

        # A pause mid-training abandons it: the network of the generation before is still on
        # disk, and the training is redone from there on resume.
        means = train_steps(self.net, self.optimizer, self.buffer, steps, self.config.batch_size, self.device, rng,
                            on_progress, before_step=lambda: self._check_pause("now"))
        self.step += steps
        return {**means, "steps": steps}

    def _evaluate(self, generation: int) -> dict:
        n = self.config.eval_games
        if n <= 0:
            return {}
        me = str(self.checkpoint(generation))
        specs = {}
        for spec in self.config.eval_opponents:
            if spec == "previous":
                specs[spec] = str(self.checkpoint(generation - 1))
            elif spec in ("greedy", "random"):
                specs[spec] = spec
        if not specs:
            return {}
        seed = self.config.seed * 1_000_003 + generation * 10_000 + 5_000
        # Every opponent's games together, side by side (see play_matches and match_plan).
        games = [(me, opponent, i, seed) for i in range(n) for opponent in specs.values()]
        self.set_status(state="evaluating", opponent=None, opponents=list(specs), games_done=0, games_total=len(games))
        self._phase(sum(n * self._eval_plies.get(spec, 220.0) for spec in specs))
        pool = self._pool_or_none()
        rows: list[dict] = []

        def keep(result: list[dict]) -> None:
            rows.extend(result)
            self.set_status(games_done=len(rows))

        if pool is None:
            for task in match_plan(games, 0, False):
                self._check_pause("now")
                keep(play_matches(task, self.config.test_simulations, self.config.eval_opening_plies))
        else:
            self._gather([pool.apply_async(play_matches, (task, self.config.test_simulations, self.config.eval_opening_plies))
                          for task in match_plan(games, self.workers, self.gpu_workers > 0)], keep)
        self._live = None
        results = {}
        for spec, opponent in specs.items():
            mine = [r for r in rows if r["opponent"] == opponent]
            self._eval_plies[spec] = sum(r["plies"] for r in mine) / len(mine)
            wins = sum(r["score"] == 1 for r in mine)
            draws = sum(r["score"] == 0.5 for r in mine)
            score = sum(r["score"] for r in mine) / len(mine)
            reasons: dict = {}
            for r in mine:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            results[spec] = {"wins": wins, "draws": draws, "losses": len(mine) - wins - draws,
                             "score": score, "reasons": reasons}
            self.record("eval", generation=generation, opponent=spec, **results[spec])
            self.log(f"  vs {spec:<8} {wins} W / {draws} D / {len(mine) - wins - draws} L  (score {score:.0%})")
        return results

    def _update_best(self, generation: int, evaluation: dict) -> None:
        """The best network scores highest against the fixed baselines (Greedy and Random, averaged:
        40 games rather than 20, so a few lucky draws do not decide it); later generations win ties.
        Without a baseline test, the newest network is the best."""
        baselines = [evaluation[k]["score"] for k in ("greedy", "random") if k in (evaluation or {})]
        score = sum(baselines) / len(baselines) if baselines else None
        if score is None or score >= self.best_score:
            self.best, self.best_score = generation, (score if score is not None else self.best_score)
        best_path = self.dir / "checkpoints" / "best.pt"
        shutil.copyfile(self.checkpoint(self.best), best_path)
        if self.publish:
            Path(self.publish).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(self.publish + ".tmp")
            shutil.copyfile(best_path, tmp)
            tmp.replace(self.publish)

    def _save_train_state(self, generation: int, evaluated: bool, steps: int = 0, losses: Optional[dict] = None) -> None:
        tmp = self.dir / "checkpoints" / "train_state.pt.tmp"
        torch.save({"generation": generation, "optimizer": self.optimizer.state_dict(), "step": self.step,
                    "best": self.best, "best_score": self.best_score, "evaluated": evaluated,
                    "steps": steps, "losses": losses or {}}, tmp)
        tmp.replace(self.dir / "checkpoints" / "train_state.pt")

    def _finish_generation(self, generation: int, steps: int, losses: dict, started: Optional[float]) -> None:
        """Evaluate a trained generation, choose the best network and close the generation. Its
        seconds run from `started` (perf_counter) to here, its test games included (until generation
        75 of main-v4 they stopped before them); 0 for a generation whose start was in another session."""
        evaluation = self._evaluate(generation)
        self._update_best(generation, evaluation)
        self._save_train_state(generation, evaluated=True)
        seconds = time.perf_counter() - started if started is not None else 0.0
        self.record("generation", generation=generation, best=self.best, steps=steps, seconds=round(seconds, 1),
                    **{k: round(v, 5) for k, v in losses.items()})
        self.set_status(state="generation_done", generation=generation, best_generation=self.best,
                        games_done=None, games_total=None, opponent=None, moves=None, moves_expected=None,
                        phase_started=None)

    def _report(self) -> None:
        """The run report, made as the run finishes (auto_report). A run already on the rating ladder
        rates its new generations there too, with the ladder's own settings."""
        from .analysis import format_strength, make_report, measure_strength
        from .benchmark import format_benchmark, read_benchmark, run_benchmark

        def part(name: str) -> Callable[[int, int], None]:
            """Show the report's part under way, and its games as they finish (as test games do)."""
            self.set_status(report_part=name, games_done=None, games_total=None, phase_started=_now())
            return lambda done, total: self.set_status(games_done=done, games_total=total)

        self.set_status(state="reporting")
        self._phase()
        self.log("Making the run report...")
        part("report")
        make_report(self.dir)
        if self.config.report_games > 0:
            data = measure_strength(self.dir, games=self.config.report_games, workers=self.workers,
                                    gpu_workers=self.gpu_workers, log=lambda *_: None, on_progress=part("strength"))
            self.log(format_strength(data))
        ladder = read_benchmark(self.dir)
        if ladder:
            data = run_benchmark(self.dir, games=ladder.get("games", 40), every=ladder.get("every", 10),
                                 workers=self.workers, gpu_workers=self.gpu_workers, log=lambda *_: None,
                                 on_progress=part("ladder"))
            self.log(format_benchmark(data))
        self.set_status(report_part=None)

    # -- the loop

    def run(self) -> None:
        global _progress
        cfg = self.config
        request_pause(self.dir, None)  # a request left over from an earlier run no longer applies
        # Games played in this process (no worker pool) count towards the same live progress.
        _progress = (self._moves, self._games)
        stop_ticking = threading.Event()

        def tick() -> None:
            while not stop_ticking.wait(0.5):
                report = self._live
                if report:
                    try:
                        report()
                    except Exception:  # progress is a courtesy; never let it stop the run
                        pass

        threading.Thread(target=tick, name="progress", daemon=True).start()
        try:
            last = self._load_or_create()
            # The settings each start or resume trains with, so changes between generations are on record.
            self.record("start", after_generation=last, config=cfg.to_json(), workers=self.workers,
                        gpu_workers=self.gpu_workers)
            self.log(f"Run {self.name}: generation {last} of {cfg.generations}, {self.buffer.games} games "
                     f"({len(self.buffer):,} positions) in the buffer; training on {self.device}, "
                     f"self-play on {self.workers or 1} {'processes' if self.workers else 'process'}"
                     + (f" ({self.gpu_workers} with the network on the GPU)" if self.gpu_workers else ""))
            if last == 0 and not (self.dir / "checkpoints" / "best.pt").exists():
                self._update_best(0, {})
            self.set_status(generation=last, best_generation=self.best, paused_in=None, paused_after=None)
            if self._pending_eval is not None:
                self.log(f"Generation {last}: evaluating (interrupted before)")
                self._finish_generation(last, self._pending_eval.get("steps") or 0, self._pending_eval["losses"], None)
                self._pending_eval = None
                self._check_pause("generation", after="generation")
            for generation in range(last + 1, cfg.generations + 1):
                t0 = time.perf_counter()
                records = self._selfplay(generation)
                for r in records:
                    self.buffer.add(r)
                summary = summarise(records)
                contributed = self._take_incoming()
                # Measured on games from the standard setup: the position the agent is for.
                held_out = held_out_loss(self.net, [r for r in records if not r.start_fen] or records, self.device,
                                         seed=generation)
                t1 = time.perf_counter()
                self.record("selfplay", generation=generation, seconds=round(t1 - t0, 1), **summary,
                            contributed=contributed["added"], rejected=contributed["rejected"], buffer_games=self.buffer.games,
                            buffer_positions=len(self.buffer))
                self.record("held_out", generation=generation, step=self.step,
                            **{k: round(v, 5) for k, v in held_out.items()})
                ends = summary.get("endgames")
                endgame_note = (f"; {ends['games']} endgame starts, {ends['games'] - ends['draws']} decisive, "
                                f"{ends['mean_plies']:.0f} plies on average") if ends else ""
                self.log(f"Generation {generation}: {len(records)} games in {t1 - t0:.0f} s; from the opening "
                         f"W {summary['white_wins']} / D {summary['draws']} / B {summary['black_wins']}, "
                         f"{summary['mean_plies']:.0f} plies on average{endgame_note}")
                self._check_pause("phase", after="selfplay")

                losses = self._train(generation, summary["positions"] + contributed["positions"])
                steps = losses.pop("steps")
                save_checkpoint(str(self.checkpoint(generation)), self.net, generation=generation, run=self.name)
                self._save_train_state(generation, evaluated=False, steps=steps, losses=losses)
                t2 = time.perf_counter()
                self.log(f"  trained {steps} steps in {t2 - t1:.0f} s: loss {losses['loss']:.3f} "
                         f"(value {losses['value_loss']:.3f}, policy {losses['policy_loss']:.3f})")
                self._check_pause("phase", after="training")

                self._finish_generation(generation, steps, losses, t0)
                if generation < cfg.generations:
                    self._check_pause("generation", after="generation")
            self._stop_pool(abandon=False)
            self.log(f"Done: {cfg.generations} generations; best network is generation {self.best}.")
            if cfg.auto_report:
                self._report()
            self.set_status(state="finished")
        except (Paused, KeyboardInterrupt):
            self._stop_pool(abandon=True)
            phase = self.status.get("state")
            after = self._paused_after
            during = None if after else (phase if phase in ("selfplay", "training", "evaluating") else None)
            self.set_status(state="paused", paused_in=during, paused_after=after)
            g = self.status.get("generation")
            names = {"selfplay": "self-play", "training": "training", "evaluating": "testing"}
            where = (f"after generation {g}" if after == "generation" else
                     f"after generation {g}'s {names[after]}" if after else
                     f"during generation {g}'s {names[during]}" if during else f"in generation {g}")
            self.log(f"Paused {where}. Run the same command again to resume.")
        except BaseException:
            self._stop_pool(abandon=True)
            self.set_status(state="failed")
            raise
        finally:
            stop_ticking.set()
            self._live = None
            _progress = None
            request_pause(self.dir, None)
