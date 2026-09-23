"""Baseline players.

Used as the "Computer" opponent in the web app before the AlphaZero agent exists, and as
benchmark opponents in Milestone 4. They use hand-written material values, which is
fine for baselines. The AlphaZero agent itself gets no such knowledge.
"""

import random
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


AGENTS = {agent.id: agent for agent in (RandomAgent, GreedyAgent)}


def make_agent(agent_id: str, seed: Optional[int] = None):
    if agent_id not in AGENTS:
        raise ValueError(f"unknown agent {agent_id!r}; choose from {sorted(AGENTS)}")
    return AGENTS[agent_id](seed)
