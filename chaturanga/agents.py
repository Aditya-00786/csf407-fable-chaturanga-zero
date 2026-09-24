"""Players: baselines and the MCTS agent.

Random and Greedy are the "Computer" opponents in the web app and the benchmark opponents for
Milestone 4. Greedy uses hand-written material values, which is fine for a baseline. The MCTS
agent gets no such knowledge: only the rules, the results of finished games, and its network.
"""

import random
import threading
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


_evaluator = None
_evaluator_lock = threading.Lock()


def untrained_evaluator():
    """One untrained network (fixed seed, so every run plays the same), shared by all MCTS agents.

    Built on first use; a caller arriving while another thread builds it waits for that build.
    """
    global _evaluator
    with _evaluator_lock:
        if _evaluator is None:
            from .net import NetworkEvaluator, new_network

            _evaluator = NetworkEvaluator(new_network(seed=0))
    return _evaluator


def warm_up() -> None:
    """Load PyTorch and the network and run one evaluation, so the first MCTS move is as fast as
    the rest. Importing PyTorch alone takes a few seconds."""
    untrained_evaluator().evaluate(GameState.initial())


class MCTSAgent:
    id = "mcts"
    name = "MCTS (untrained)"
    description = ("Searches 50 positions ahead for every move, guided by a network that has not been trained "
                   "yet: it knows only the rules and how games end.")

    def __init__(self, seed: Optional[int] = None, simulations: int = 50, evaluator=None):
        from .mcts import MCTS, MCTSConfig

        # No root noise and the most visited move: this agent plays to win, it is not exploring.
        self.mcts = MCTS(evaluator or untrained_evaluator(), MCTSConfig(simulations=simulations), seed)

    def select_move(self, state: GameState) -> int:
        root = self.mcts.search(state)
        return self.mcts.choose_move(root, temperature=0.0)


AGENTS = {agent.id: agent for agent in (RandomAgent, GreedyAgent, MCTSAgent)}


def make_agent(agent_id: str, seed: Optional[int] = None):
    if agent_id not in AGENTS:
        raise ValueError(f"unknown agent {agent_id!r}; choose from {sorted(AGENTS)}")
    return AGENTS[agent_id](seed)
