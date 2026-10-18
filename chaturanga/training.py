"""The training loop (Milestone 3): self-play, train, evaluate, repeat.

Each generation g:
  1. self-play: the network of generation g-1 plays `games_per_generation` games against itself;
  2. teammates' games waiting in `incoming/` are checked and added (see `scripts/selfplay.py`);
  3. training: batches from the replay buffer (enough to sample each new position about
     `sample_reuse` times, at most `steps_per_generation`), giving network g;
  4. evaluation: network g plays network g-1 and the Greedy baseline.

Self-play always uses the newest network, as in AlphaZero (no gating). The best network, the one
that scores highest against Greedy (later generations win ties), is copied to `best.pt` and,
when asked, published where the web app picks it up.

Everything a run produces lives in its directory, so a run stops and resumes at any point:

    runs/<name>/
      config.json            the run's settings
      metrics.jsonl          one JSON object per event: training losses, self-play results, evaluations
      status.json            what the run is doing now (read by the app's Watch training view)
      games/gen_0001.jsonl.gz  self-play games of each generation (teammates' games in games/contrib/)
      incoming/              drop teammates' game files here; they are added at the next generation
      checkpoints/gen_0000.pt  one network per generation, best.pt, and train_state.pt (optimiser)
"""

import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .game import GameState
from .mcts import MCTS, MCTSConfig
from .net import NetConfig, NetworkEvaluator, PolicyValueNet, load_checkpoint, new_network, save_checkpoint
from .replay_buffer import ReplayBuffer
from .selfplay import GameRecord, SelfPlayConfig, play_games, read_games, summarise, write_games


@dataclass(frozen=True)
class TrainConfig:
    generations: int = 20             # stop after this many; raise it and resume to train further
    games_per_generation: int = 50
    simulations: int = 50             # MCTS simulations per move in self-play and evaluation
    temperature_plies: int = 30
    concurrent: int = 8               # self-play games per process, sharing batched evaluations
    buffer_positions: int = 60_000    # about the last six generations
    batch_size: int = 256
    steps_per_generation: int = 300   # at most; see sample_reuse
    # Training steps per generation are sized so the new positions are sampled about this many
    # times on average (a replay ratio). More passes over few games make the value head memorise
    # which game a position came from instead of learning to judge it. 0 always runs the maximum.
    sample_reuse: float = 4.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eval_games: int = 20              # per opponent after each generation; 0 skips evaluation
    eval_opponents: tuple = ("previous", "greedy")
    blocks: int = NetConfig.blocks
    channels: int = NetConfig.channels
    seed: int = 0

    @property
    def net_config(self) -> NetConfig:
        return NetConfig(blocks=self.blocks, channels=self.channels)

    @property
    def selfplay_config(self) -> SelfPlayConfig:
        return SelfPlayConfig(self.simulations, self.temperature_plies, self.concurrent)

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


def training_device() -> torch.device:
    """Training sends large batches, where a GPU (CUDA on Colab, MPS on a Mac) pays off."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- workers

# One evaluator per worker process, reloaded when the checkpoint changes.
_worker_cache: dict = {}


def _init_worker() -> None:
    torch.set_num_threads(1)  # one process per core is faster than threads for batch sizes this small


def _evaluator_for(checkpoint: str) -> NetworkEvaluator:
    key = (checkpoint, os.path.getmtime(checkpoint))
    if _worker_cache.get("key") != key:
        net, _ = load_checkpoint(checkpoint)
        _worker_cache.update(key=key, evaluator=NetworkEvaluator(net, torch.device("cpu")))
    return _worker_cache["evaluator"]


def selfplay_task(checkpoint: str, n_games: int, config: SelfPlayConfig, seed: int,
                  generation: int, contributor: str) -> list[dict]:
    records = play_games(_evaluator_for(checkpoint), n_games, config, seed, generation, contributor)
    return [r.to_json() for r in records]


class SearchPlayer:
    """MCTS with no noise. It samples its first `opening_plies` moves in proportion to visits,
    so that evaluation games between deterministic players still differ."""

    def __init__(self, evaluator, simulations: int, seed: int, opening_plies: int = 8):
        self.mcts = MCTS(evaluator, MCTSConfig(simulations=simulations), seed=seed)
        self.opening_plies = opening_plies

    def select_move(self, state: GameState) -> int:
        root = self.mcts.search(state)
        return self.mcts.choose_move(root, 1.0 if state.ply < self.opening_plies else 0.0)


def _opponent(spec: str, seed: int, simulations: int):
    if spec == "greedy":
        return GreedyAgent(seed)
    if spec == "random":
        return RandomAgent(seed)
    return SearchPlayer(_evaluator_for(spec), simulations, seed)  # a checkpoint path


def match_task(checkpoint: str, opponent: str, game_indices: list[int], simulations: int, seed: int) -> list[dict]:
    """Games between a checkpoint and an opponent; the checkpoint plays White in even-numbered games."""
    out = []
    for i in game_indices:
        player = SearchPlayer(_evaluator_for(checkpoint), simulations, seed + i)
        other = _opponent(opponent, seed + 7919 * (i + 1), simulations)
        colour = WHITE if i % 2 == 0 else -WHITE
        state = GameState.initial()
        while not state.is_terminal():
            mover = player if state.side == colour else other
            state = state.apply(mover.select_move(state))
        result = state.outcome()
        out.append({"score": (result.value_for(colour) + 1) / 2, "reason": result.reason, "plies": state.ply,
                    "colour": colour})
    return out


# ---------------------------------------------------------------------- training steps

def train_steps(net: PolicyValueNet, optimizer: torch.optim.Optimizer, buffer: ReplayBuffer, steps: int,
                batch_size: int, device: torch.device, rng: np.random.Generator,
                on_progress: Optional[Callable[[int, dict], None]] = None, report_every: int = 10) -> dict:
    """Run `steps` optimiser steps on batches from the buffer. The loss is AlphaZero's: the value's
    squared error plus the policy's cross-entropy against the search's visit shares (weight decay
    is in the optimiser). Returns the mean of each loss over the steps."""
    net.train()
    totals = {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0, "policy_entropy": 0.0}
    window = dict.fromkeys(totals, 0.0)
    for step in range(1, steps + 1):
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
                 publish: Optional[str] = None, log: Callable[[str], None] = print):
        self.dir = Path(directory)
        self.name = self.dir.name
        self.workers = workers
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
                             "generations": self.config.generations, "pid": os.getpid()}
        self.buffer = ReplayBuffer(self.config.buffer_positions)
        self.net: Optional[PolicyValueNet] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.step = 0
        self.best = 0
        self.best_score = -1.0
        self._pool: Optional[ProcessPoolExecutor] = None

    # -- paths and records

    def checkpoint(self, generation: int) -> Path:
        return self.dir / "checkpoints" / f"gen_{generation:04d}.pt"

    def games_file(self, generation: int) -> Path:
        return self.dir / "games" / f"gen_{generation:04d}.jsonl.gz"

    def record(self, event: str, **data) -> None:
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"event": event, "time": _now(), **data}) + "\n")

    def set_status(self, **changes) -> None:
        self.status.update(changes, updated=_now())
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status))
        tmp.replace(self.dir / "status.json")

    # -- setup and resume

    def _completed_generations(self) -> int:
        done = [int(p.stem[4:]) for p in (self.dir / "checkpoints").glob("gen_*.pt")]
        return max(done) if done else -1

    def _load_or_create(self) -> int:
        """Load the newest network (or create generation 0) and refill the buffer. Returns the
        newest completed generation."""
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
                self.step = state["step"]
            self.best, self.best_score = state.get("best", 0), state.get("best_score", -1.0)

        # Refill the buffer from the newest games backwards, then add them oldest first.
        files = sorted((self.dir / "games").glob("gen_*.jsonl.gz"), reverse=True)
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

    # -- the phases of a generation

    def _pool_or_none(self) -> Optional[ProcessPoolExecutor]:
        if self.workers <= 0:
            return None
        if self._pool is None:
            import multiprocessing as mp

            self._pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
        return self._pool

    def _selfplay(self, generation: int) -> list[GameRecord]:
        path = self.games_file(generation)
        if path.exists():  # already played before an interruption
            return list(read_games(str(path)))
        n = self.config.games_per_generation
        seed = self.config.seed * 1_000_003 + generation * 10_000
        net_path = str(self.checkpoint(generation - 1))
        records: list[GameRecord] = []
        self.set_status(state="selfplay", generation=generation, games_done=0, games_total=n)
        pool = self._pool_or_none()
        if pool is None:
            evaluator = NetworkEvaluator(self.net, self.selfplay_device)

            def on_game(r):
                records.append(r)
                self.set_status(games_done=len(records))

            play_games(evaluator, n, self.config.selfplay_config, seed, generation - 1, "", DEFAULT_RULES, on_game)
            self.net.to(self.device)
        else:
            chunk = max(1, min(self.config.concurrent, math.ceil(n / (2 * self.workers))))
            futures = [pool.submit(selfplay_task, net_path, min(chunk, n - start), self.config.selfplay_config,
                                   seed + start, generation - 1, "") for start in range(0, n, chunk)]
            for fut in as_completed(futures):
                records.extend(GameRecord.from_json(d) for d in fut.result())
                self.set_status(games_done=len(records))
            records.sort(key=lambda r: r.seed)
        tmp = path.with_suffix(".tmp")
        write_games(str(tmp), records)
        tmp.replace(path)
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
            except (OSError, ValueError, TypeError) as exc:
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
        rng = np.random.default_rng(self.config.seed * 7_919 + generation)

        def on_progress(step, losses):
            self.record("train", generation=generation, step=self.step + step, **{k: round(v, 5) for k, v in losses.items()})
            self.set_status(step=step)

        means = train_steps(self.net, self.optimizer, self.buffer, steps, self.config.batch_size, self.device, rng, on_progress)
        self.step += steps
        return {**means, "steps": steps}

    def _evaluate(self, generation: int) -> dict:
        n = self.config.eval_games
        if n <= 0:
            return {}
        results = {}
        me = str(self.checkpoint(generation))
        for spec in self.config.eval_opponents:
            if spec == "previous":
                opponent = str(self.checkpoint(generation - 1))
            elif spec in ("greedy", "random"):
                opponent = spec
            else:
                continue
            self.set_status(state="evaluating", opponent=spec, games_done=0, games_total=n)
            seed = self.config.seed * 1_000_003 + generation * 10_000 + 5_000
            pool = self._pool_or_none()
            games: list[dict] = []
            if pool is None:
                for i in range(n):
                    games.extend(match_task(me, opponent, [i], self.config.simulations, seed))
                    self.set_status(games_done=len(games))
            else:
                futures = [pool.submit(match_task, me, opponent, [i], self.config.simulations, seed) for i in range(n)]
                for fut in as_completed(futures):
                    games.extend(fut.result())
                    self.set_status(games_done=len(games))
            wins = sum(g["score"] == 1 for g in games)
            draws = sum(g["score"] == 0.5 for g in games)
            score = sum(g["score"] for g in games) / len(games)
            reasons: dict = {}
            for g in games:
                reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
            results[spec] = {"wins": wins, "draws": draws, "losses": len(games) - wins - draws,
                             "score": score, "reasons": reasons}
            self.record("eval", generation=generation, opponent=spec, **results[spec])
            self.log(f"  vs {spec:<8} {wins} W / {draws} D / {len(games) - wins - draws} L  (score {score:.0%})")
        return results

    def _update_best(self, generation: int, evaluation: dict) -> None:
        score = evaluation.get("greedy", {}).get("score", 0.0) if evaluation else 0.0
        if not evaluation or "greedy" not in evaluation or score >= self.best_score:
            self.best, self.best_score = generation, score
        best_path = self.dir / "checkpoints" / "best.pt"
        shutil.copyfile(self.checkpoint(self.best), best_path)
        if self.publish:
            Path(self.publish).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(self.publish + ".tmp")
            shutil.copyfile(best_path, tmp)
            tmp.replace(self.publish)

    def _save_train_state(self, generation: int) -> None:
        tmp = self.dir / "checkpoints" / "train_state.pt.tmp"
        torch.save({"generation": generation, "optimizer": self.optimizer.state_dict(), "step": self.step,
                    "best": self.best, "best_score": self.best_score}, tmp)
        tmp.replace(self.dir / "checkpoints" / "train_state.pt")

    # -- the loop

    def run(self) -> None:
        cfg = self.config
        try:
            last = self._load_or_create()
            self.log(f"Run {self.name}: generation {last} of {cfg.generations}, {self.buffer.games} games "
                     f"({len(self.buffer):,} positions) in the buffer; training on {self.device}, "
                     f"self-play on {self.workers or 1} {'processes' if self.workers else 'process'}")
            if last == 0 and not (self.dir / "checkpoints" / "best.pt").exists():
                self._update_best(0, {})
            for generation in range(last + 1, cfg.generations + 1):
                t0 = time.perf_counter()
                records = self._selfplay(generation)
                for r in records:
                    self.buffer.add(r)
                summary = summarise(records)
                contributed = self._take_incoming()
                held_out = held_out_loss(self.net, records, self.device, seed=generation)
                t1 = time.perf_counter()
                self.record("selfplay", generation=generation, seconds=round(t1 - t0, 1), **summary,
                            contributed=contributed["added"], rejected=contributed["rejected"], buffer_games=self.buffer.games,
                            buffer_positions=len(self.buffer))
                self.record("held_out", generation=generation, step=self.step,
                            **{k: round(v, 5) for k, v in held_out.items()})
                self.log(f"Generation {generation}: {summary['games']} games in {t1 - t0:.0f} s "
                         f"(W {summary['white_wins']} / D {summary['draws']} / B {summary['black_wins']}, "
                         f"{summary['mean_plies']:.0f} plies on average)")

                losses = self._train(generation, summary["positions"] + contributed["positions"])
                steps = losses.pop("steps")
                save_checkpoint(str(self.checkpoint(generation)), self.net, generation=generation, run=self.name)
                self._save_train_state(generation)
                t2 = time.perf_counter()
                self.log(f"  trained {steps} steps in {t2 - t1:.0f} s: loss {losses['loss']:.3f} "
                         f"(value {losses['value_loss']:.3f}, policy {losses['policy_loss']:.3f})")

                evaluation = self._evaluate(generation)
                self._update_best(generation, evaluation)
                self._save_train_state(generation)
                self.record("generation", generation=generation, best=self.best, steps=steps,
                            seconds=round(time.perf_counter() - t0, 1),
                            **{k: round(v, 5) for k, v in losses.items()})
                self.set_status(state="generation_done", generation=generation, best_generation=self.best)
            self.set_status(state="finished")
            self.log(f"Done: {cfg.generations} generations; best network is generation {self.best}.")
        except KeyboardInterrupt:
            self.set_status(state="stopped")
            self.log("Stopped. Run the same command again to resume.")
        except BaseException:
            self.set_status(state="failed")
            raise
        finally:
            if self._pool is not None:
                self._pool.shutdown(cancel_futures=True)
