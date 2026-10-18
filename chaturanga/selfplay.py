"""Self-play: the network plays itself with MCTS, and every move becomes a training example.

Several games are played side by side, in lockstep: each round runs one simulation in every
game and evaluates all their new leaves in a single batch, which is what makes a GPU worth
having. On a CPU the batching still saves a little; the main speed-up there is running one
process per core (see `training.py`).

A game is recorded as plain data (`GameRecord`): the moves, the search's visit counts at every
move, and the result. The training examples are rebuilt from it by replaying the moves, so a
record is small, readable, and can be shared: a teammate's self-play games are just more records.
"""

import gzip
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from .config import DEFAULT_RULES, RulesConfig
from .encoding import ACTION_SIZE, encode_state, move_to_action
from .game import GameState, move_to_str, parse_move
from .mcts import MCTS, MCTSConfig, Evaluator


@dataclass(frozen=True)
class SelfPlayConfig:
    simulations: int = 50        # per move (the proposal's 30-50)
    temperature_plies: int = 30  # sample moves in proportion to visits for this many plies...
    concurrent: int = 8          # games played side by side, sharing batched evaluations
    noise: bool = True           # Dirichlet noise on the root priors, so self-play explores
    late_temperature: float = 0.0  # ...and after them at this temperature (0: always the most visited move)


@dataclass
class GameRecord:
    """One self-play game. `visits[i]` holds the search's visit counts before `moves[i]`."""
    moves: list[str]
    visits: list[dict[str, int]]
    winner: Optional[int]        # 1 White, -1 Black, None for a draw
    reason: str
    generation: int = 0          # the network that played it
    contributor: str = ""
    seed: Optional[int] = None
    rules: dict = field(default_factory=lambda: asdict(DEFAULT_RULES))

    @property
    def plies(self) -> int:
        return len(self.moves)

    def to_json(self) -> dict:
        return {**asdict(self), "plies": self.plies}

    @classmethod
    def from_json(cls, data: dict) -> "GameRecord":
        data = dict(data)
        data.pop("plies", None)
        return cls(**data)


# ---------------------------------------------------------------------- playing

class _Game:
    def __init__(self, index: int, evaluator: Evaluator, mcts_config: MCTSConfig, seed: int, rules: RulesConfig):
        self.index = index
        self.seed = seed
        self.mcts = MCTS(evaluator, mcts_config, seed=seed)
        self.state = GameState.initial(rules)
        self.moves: list[str] = []
        self.visits: list[dict[str, int]] = []
        self.root = None


def play_games(evaluator: Evaluator, n_games: int, config: SelfPlayConfig = SelfPlayConfig(),
               seed: int = 0, generation: int = 0, contributor: str = "",
               rules: RulesConfig = DEFAULT_RULES,
               on_game: Optional[Callable[[GameRecord], None]] = None) -> list[GameRecord]:
    """Play `n_games` self-play games, `config.concurrent` at a time. Game i uses seed + i, so the
    same seed and network always produce the same games."""
    mcts_config = MCTSConfig(simulations=config.simulations)
    waiting = list(range(n_games))
    active: list[_Game] = []
    done: list[GameRecord] = []
    while waiting or active:
        while waiting and len(active) < max(1, config.concurrent):
            i = waiting.pop(0)
            active.append(_Game(i, evaluator, mcts_config, seed + i, rules))

        for g in active:
            g.root = g.mcts.prepare(g.state, add_noise=config.noise)
        for _ in range(config.simulations):
            paths = [(g, p) for g in active if (p := g.mcts.descend(g.root)) is not None]
            if paths:
                results = evaluator.evaluate_many([p[-1].state for _, p in paths])
                for (g, path), (priors, value) in zip(paths, results):
                    g.mcts.expand_and_backup(path, priors, value)

        still = []
        for g in active:
            counts = MCTS.visit_counts(g.root)
            g.visits.append({move_to_str(m): c for m, c in counts.items() if c})
            temperature = 1.0 if g.state.ply < config.temperature_plies else config.late_temperature
            move = g.mcts.choose_move(g.root, temperature)
            g.moves.append(move_to_str(move))
            g.state = g.state.apply(move)
            if g.state.is_terminal():
                out = g.state.outcome()
                record = GameRecord(g.moves, g.visits, out.winner, out.reason, generation, contributor,
                                    g.seed, asdict(rules))
                done.append(record)
                if on_game:
                    on_game(record)
            else:
                still.append(g)
        active = still
    return done


# ---------------------------------------------------------------------- training examples

def examples(record: GameRecord) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Replay a record into training examples, one per position:
    planes (n, 15, 8, 8), and for each position the canonical actions searched with their visit
    shares (the policy target, sparse), and the result for the side to move (the value target).
    Raises ValueError if the record is not a legal game under its rules."""
    rules = RulesConfig(**record.rules)
    state = GameState.initial(rules)
    if len(record.visits) != len(record.moves):
        raise ValueError("visits and moves differ in length")
    planes, actions, shares, sides = [], [], [], []
    for text, visits in zip(record.moves, record.visits):
        if state.is_terminal():
            raise ValueError("move played after the game ended")
        move = parse_move(text)
        if not state.is_legal(move):
            raise ValueError(f"illegal move {text} in {state.to_fen()}")
        a, p = policy_target(state, visits)
        planes.append(encode_state(state))
        actions.append(a)
        shares.append(p)
        sides.append(state.side)
        state = state.apply(move)
    out = state.outcome()
    if out is None:
        raise ValueError("the game is not finished")
    if (out.winner, out.reason) != (record.winner, record.reason):
        raise ValueError(f"recorded result {record.winner}/{record.reason} but the moves give {out.winner}/{out.reason}")
    z = np.array([out.value_for(s) for s in sides], dtype=np.float32)
    return np.stack(planes) if planes else np.zeros((0, 15, 8, 8), np.float32), actions, shares, z


def policy_target(state: GameState, visits: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """The policy target from recorded visit counts, sparse: canonical actions and their shares
    (MCTS.policy_target is the same thing, dense). Raises ValueError for counts that do not fit."""
    legal = set(state.legal_moves())
    searched = [(parse_move(m), c) for m, c in visits.items()]
    total = sum(c for _, c in searched)
    if not searched or total <= 0 or any(c < 0 or m not in legal for m, c in searched):
        raise ValueError(f"bad visit counts in {state.to_fen()}")
    actions = np.array([move_to_action(m, state.side, state.rules.setup) for m, _ in searched], dtype=np.int16)
    return actions, np.array([c / total for _, c in searched], dtype=np.float32)


def dense_policy(actions: np.ndarray, shares: np.ndarray) -> np.ndarray:
    pi = np.zeros(ACTION_SIZE, dtype=np.float32)
    pi[actions.astype(np.int64)] = shares
    return pi


# ---------------------------------------------------------------------- files

def write_games(path: str, records: Iterable[GameRecord]) -> int:
    """Write records as gzipped JSON lines (one game per line). Returns how many were written."""
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_games(path: str) -> Iterator[GameRecord]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield GameRecord.from_json(json.loads(line))


def summarise(records: list[GameRecord]) -> dict:
    """Results, endings and lengths of a set of games, for the training log."""
    n = len(records)
    white = sum(r.winner == 1 for r in records)
    black = sum(r.winner == -1 for r in records)
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    plies = [r.plies for r in records] or [0]
    return {"games": n, "white_wins": white, "black_wins": black, "draws": n - white - black,
            "reasons": reasons, "mean_plies": float(np.mean(plies)), "positions": int(sum(plies))}

