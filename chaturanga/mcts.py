"""PUCT Monte Carlo Tree Search guided by a policy-value evaluator, as in AlphaZero.

Each simulation walks down the tree choosing the child with the highest

    Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

where P is the evaluator's prior, N the visit counts and Q the mean value of the child from the
point of view of the player choosing the move. At the first unexpanded node the evaluator gives
priors for its moves and a value for its side to move (no random rollouts). Finished games are
scored exactly by the engine instead. The value is then backed up the path, changing sign at
every level because the players alternate.

The evaluator is any object with `evaluate(state) -> (priors, value)`, where priors maps each
legal move to a probability and value is the expected result for the side to move, in [-1, 1].
`net.NetworkEvaluator` is the real one; `UniformEvaluator` knows only the rules.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .encoding import ACTION_SIZE, move_to_action
from .game import GameState


class Evaluator(Protocol):
    def evaluate(self, state: GameState) -> tuple[dict[int, float], float]: ...


class UniformEvaluator:
    """Equal priors for every legal move and a neutral value. With it, the search is guided by
    nothing but the rules and the results of finished games."""

    def evaluate(self, state: GameState) -> tuple[dict[int, float], float]:
        moves = state.legal_moves()
        return {m: 1.0 / len(moves) for m in moves}, 0.0


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 50          # the proposal's 30-50 per move during training
    c_puct: float = 1.5            # exploration weight
    dirichlet_alpha: float = 0.3   # root noise for self-play (AlphaZero used 0.3 for chess)
    dirichlet_fraction: float = 0.25  # share of the root prior replaced by noise when enabled
    fpu: float = 0.0               # value assumed for a move that has not been tried yet


class Node:
    """A position in the tree, reached by one move from its parent.

    `visits` and `value_sum` are seen from the player who made that move, so a parent picks the
    child with the best mean value without any sign juggling.
    """

    __slots__ = ("state", "prior", "children", "visits", "value_sum")

    def __init__(self, prior: float, state: Optional[GameState] = None):
        self.state = state        # created when the node is first visited
        self.prior = prior
        self.children: Optional[dict[int, "Node"]] = None  # move -> child; {} once found terminal
        self.visits = 0
        self.value_sum = 0.0

    @property
    def expanded(self) -> bool:
        return self.children is not None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    """A search tree that persists between moves: searching a position that was already in the
    tree (the reply to our last move, say) keeps the work done on it."""

    def __init__(self, evaluator: Evaluator, config: MCTSConfig = MCTSConfig(), seed: Optional[int] = None):
        self.evaluator = evaluator
        self.config = config
        self.np_rng = np.random.default_rng(seed)
        self.rng = random.Random(seed)
        self.root: Optional[Node] = None

    def reset(self) -> None:
        self.root = None

    # ------------------------------------------------------------------ search

    def search(self, state: GameState, add_noise: bool = False, simulations: Optional[int] = None) -> Node:
        """Run the simulations from `state` and return the root, whose children hold the results."""
        if state.is_terminal():
            raise ValueError("cannot search a finished game")
        root = self._root_for(state)
        if not root.expanded:
            value = self._expand(root)
            root.visits += 1
            root.value_sum -= value  # root is seen from the player who moved into it
        if add_noise and self.config.dirichlet_fraction > 0:
            self._add_noise(root)
        for _ in range(self.config.simulations if simulations is None else simulations):
            self._simulate(root)
        return root

    def _simulate(self, root: Node) -> None:
        node, path = root, [root]
        while node.children:  # expanded and not finished
            move, child = self._select(node)
            if child.state is None:
                child.state = node.state.apply(move)
            node = child
            path.append(node)
        outcome = node.state.outcome()
        if outcome is not None:
            node.children = {}
            value = float(outcome.value_for(node.state.side))
        else:
            value = self._expand(node)
        # `value` is for the side to move at the leaf; each node stores it for the player who
        # moved into it, which alternates on the way up.
        for n in reversed(path):
            value = -value
            n.visits += 1
            n.value_sum += value

    def _select(self, node: Node) -> tuple[int, Node]:
        c, fpu = self.config.c_puct, self.config.fpu
        scale = c * math.sqrt(node.visits)
        best_move, best_child, best_score = None, None, -math.inf
        for move, child in node.children.items():
            q = child.value_sum / child.visits if child.visits else fpu
            score = q + scale * child.prior / (1 + child.visits)
            if score > best_score:
                best_move, best_child, best_score = move, child, score
        return best_move, best_child

    def _expand(self, node: Node) -> float:
        priors, value = self.evaluator.evaluate(node.state)
        node.children = {m: Node(p) for m, p in priors.items()}
        return value

    def _add_noise(self, root: Node) -> None:
        children = list(root.children.values())
        noise = self.np_rng.dirichlet([self.config.dirichlet_alpha] * len(children))
        eps = self.config.dirichlet_fraction
        for child, n in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _root_for(self, state: GameState) -> Node:
        """Reuse a matching node from the current tree (root, child or grandchild), else start fresh."""
        if self.root is not None:
            level = [self.root]
            for _ in range(3):
                for node in level:
                    if node.state is not None and _same_position(node.state, state):
                        self.root = node
                        return node
                level = [c for n in level for c in (n.children or {}).values()]
        self.root = Node(1.0, state)
        return self.root

    # ------------------------------------------------------------------ results

    @staticmethod
    def visit_counts(root: Node) -> dict[int, int]:
        return {m: c.visits for m, c in root.children.items()}

    def choose_move(self, root: Node, temperature: float = 0.0) -> int:
        """Pick a move from the visit counts: the most visited one at temperature 0 (ties broken
        at random), otherwise with probability proportional to visits^(1/temperature)."""
        counts = self.visit_counts(root)
        moves = list(counts)
        if temperature <= 0:
            top = max(counts.values())
            return self.rng.choice([m for m in moves if counts[m] == top])
        weights = np.array([counts[m] for m in moves], dtype=np.float64) ** (1.0 / temperature)
        if weights.sum() == 0:
            weights = np.ones(len(moves))
        return moves[self.np_rng.choice(len(moves), p=weights / weights.sum())]

    @staticmethod
    def policy_target(root: Node) -> np.ndarray:
        """The visit distribution over the 4,096 canonical actions: the policy training target."""
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        state = root.state
        total = sum(c.visits for c in root.children.values())
        for move, child in root.children.items():
            pi[move_to_action(move, state.side, state.rules.setup)] = child.visits / total if total else 0.0
        return pi

    @staticmethod
    def root_value(root: Node) -> float:
        """The search's estimate of the result for the side to move at the root."""
        return -root.mean_value


def _same_position(a: GameState, b: GameState) -> bool:
    return (a.hash == b.hash and a.ply == b.ply and a.no_progress == b.no_progress
            and a.board == b.board and list(a.history) == list(b.history))
