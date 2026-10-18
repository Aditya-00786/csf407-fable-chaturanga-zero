"""Players: baselines and the MCTS agent.

Random and Greedy are the "Computer" opponents in the web app and the benchmark opponents for
Milestone 4. Greedy uses hand-written material values, which is fine for a baseline. The MCTS
agent gets no such knowledge: only the rules, the results of finished games, and its network.
"""

import errno
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import ASHVA, GAJA, MANTRI, PADATI, RAJA, RATHA
from .game import GameState

# Rough shatranj-era values (al-Adli via Murray pp. 225-226): Ratha strongest,
# Ashva next, Gaja and Mantri weak, Padati weakest.
PIECE_VALUES = {PADATI: 1.0, ASHVA: 3.0, GAJA: 1.5, MANTRI: 1.5, RATHA: 5.0, RAJA: 0.0}

WIN_SCORE = 1000.0


def material_balance(state: GameState, color: int) -> float:
    """Material of `color` minus material of the opponent."""
    total = 0.0
    for p in state.board:
        if p:
            total += PIECE_VALUES[abs(p)] if (p > 0) == (color > 0) else -PIECE_VALUES[abs(p)]
    return total


class RandomAgent:
    id = "random"
    name = "Random"
    description = "Plays a uniformly random legal move."

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, state: GameState) -> int:
        return self.rng.choice(state.legal_moves())


class GreedyAgent:
    id = "greedy"
    name = "Greedy"
    description = "Looks two plies ahead: grabs material and wins, avoids losing pieces to the reply."

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, state: GameState) -> int:
        me = state.side
        best_score, best_moves = None, []
        for move in state.legal_moves():
            score = self._score_after(state.apply(move), me)
            if best_score is None or score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return self.rng.choice(best_moves)

    def _score_after(self, child: GameState, me: int) -> float:
        """Score for `me` after my move, assuming the opponent replies as well as possible."""
        out = child.outcome()
        if out is not None:
            return WIN_SCORE * out.value_for(me)
        worst = None
        for reply in child.legal_moves():
            grandchild = child.apply(reply)
            out = grandchild.outcome()
            score = WIN_SCORE * out.value_for(me) if out is not None else material_balance(grandchild, me)
            if worst is None or score < worst:
                worst = score
        return worst


class NetworkUnavailable(RuntimeError):
    """The MCTS network is not available: it failed to load, or is still loading."""


# How long an MCTS move waits for the network while another thread is still loading it.
NETWORK_WAIT_SECONDS = 60.0

_evaluator = None
_load_error: Optional[str] = None
_evaluator_lock = threading.Lock()


def _describe(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.errno == errno.ETIMEDOUT:
        return ("reading PyTorch's files from disk timed out. If the project is in a folder synced by "
                "iCloud Drive, macOS may have offloaded them; keep the folder downloaded")
    return f"{type(exc).__name__}: {exc}"


def untrained_evaluator(timeout: Optional[float] = None):
    """One untrained network (fixed seed, so every run plays the same), shared by all MCTS agents.

    Built on first use. A caller arriving while another thread builds it waits for that build, for
    at most `timeout` seconds (None: as long as it takes). Raises NetworkUnavailable if the build
    failed or the wait ran out.

    A failed load is final for the process: a half-imported PyTorch cannot be imported again (it
    fails registering its operators twice), so later calls report the first failure instead.
    """
    global _evaluator, _load_error
    if _evaluator is not None:
        return _evaluator
    if not _evaluator_lock.acquire(timeout=-1 if timeout is None else timeout):
        raise NetworkUnavailable("The MCTS network is still loading. Try again in a moment.")
    try:
        if _evaluator is None and _load_error is None:
            try:
                from .net import NetworkEvaluator, new_network

                _evaluator = NetworkEvaluator(new_network(seed=0))
            except Exception as exc:
                _load_error = _describe(exc)
        if _evaluator is None:
            raise NetworkUnavailable(f"The MCTS network could not be loaded: {_load_error}. "
                                     "Restart the API server to try again.")
        return _evaluator
    finally:
        _evaluator_lock.release()


def trained_checkpoint() -> Path:
    """Where the trained network is published (scripts/train.py copies its best network here)."""
    default = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"
    return Path(os.environ.get("CHATURANGA_CHECKPOINT", default))


_trained: dict = {}
_trained_lock = threading.Lock()


def trained_evaluator():
    """(evaluator, info) for the published trained network, or None if there is none yet.

    The file is checked on every call and reloaded when it changes, so while a training run
    publishes better networks the app plays the newest without a restart."""
    path = trained_checkpoint()
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    with _trained_lock:
        if _trained.get("key") != key:
            untrained_evaluator(timeout=NETWORK_WAIT_SECONDS)  # PyTorch itself loads (and fails) there
            from .net import NetworkEvaluator, load_checkpoint

            try:
                net, meta = load_checkpoint(str(path))
            except Exception as exc:  # a half-written or foreign file: keep what we had
                if "evaluator" in _trained:
                    return _trained["evaluator"], _trained["info"]
                raise NetworkUnavailable(f"The trained network could not be loaded: {_describe(exc)}") from exc
            info = {"run": meta.get("run"), "generation": meta.get("generation")}
            _trained.update(key=key, evaluator=NetworkEvaluator(net), info=info)
        return _trained["evaluator"], _trained["info"]


def warm_up() -> None:
    """Load PyTorch and the networks and run one evaluation, so the first MCTS move is as fast as
    the rest. Importing PyTorch alone takes a few seconds."""
    untrained_evaluator().evaluate(GameState.initial())
    trained = trained_evaluator()
    if trained:
        trained[0].evaluate(GameState.initial())


class MCTSAgent:
    id = "mcts"
    name = "MCTS (untrained)"
    description = ("Searches 50 positions ahead for every move, guided by a network that has not been trained "
                   "yet: it knows only the rules and how games end.")

    def __init__(self, seed: Optional[int] = None, simulations: int = 50, evaluator=None, temperature: float = 0.0):
        from .mcts import MCTS, MCTSConfig

        # No root noise: this agent plays to win, it is not exploring. At temperature 0 it plays
        # the most visited move; above 0 it samples in proportion to visits (weaker, more varied).
        if evaluator is None:
            evaluator = untrained_evaluator(timeout=NETWORK_WAIT_SECONDS)
        self.mcts = MCTS(evaluator, MCTSConfig(simulations=simulations), seed)
        self.temperature = temperature

    def select_move(self, state: GameState) -> int:
        root = self.mcts.search(state)
        return self.mcts.choose_move(root, temperature=self.temperature)


@dataclass(frozen=True)
class Level:
    id: str
    name: str
    simulations: int
    temperature: float
    description: str


# Strength comes from the search budget and from how often the agent strays from its best move.
LEVELS = {level.id: level for level in (
    Level("easy", "Easy", 10, 1.0, "A short search, and it often plays a move other than its favourite."),
    Level("medium", "Medium", 50, 0.5, "The training search; it mostly plays its favourite move."),
    Level("hard", "Hard", 200, 0.0, "Four times the training search, always its best move."),
    Level("impossible", "Impossible", 800, 0.0, "Sixteen times the training search; thinks for about a second."),
)}
DEFAULT_LEVEL = "medium"


class ZeroAgent(MCTSAgent):
    id = "zero"
    name = "ChaturangZero"
    description = "MCTS guided by the network trained by self-play, at a chosen level."

    def __init__(self, seed: Optional[int] = None, level: str = DEFAULT_LEVEL):
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; choose from {list(LEVELS)}")
        trained = trained_evaluator()
        if trained is None:
            raise NetworkUnavailable("There is no trained network yet. Train one with scripts/train.py "
                                     "(it publishes checkpoints/best.pt).")
        evaluator, self.network = trained
        self.level = LEVELS[level]
        super().__init__(seed, self.level.simulations, evaluator, self.level.temperature)


AGENTS = {agent.id: agent for agent in (RandomAgent, GreedyAgent, MCTSAgent, ZeroAgent)}


def make_agent(agent_id: str, seed: Optional[int] = None, level: Optional[str] = None):
    if agent_id not in AGENTS:
        raise ValueError(f"unknown agent {agent_id!r}; choose from {sorted(AGENTS)}")
    if agent_id == ZeroAgent.id:
        return ZeroAgent(seed, level or DEFAULT_LEVEL)
    return AGENTS[agent_id](seed)
