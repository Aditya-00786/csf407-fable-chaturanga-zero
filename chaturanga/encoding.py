"""Tensor encoding of states and moves for the neural network.

Everything is expressed from the point of view of the side to move ("canonical" form),
so the network always plays as if it were White:
  - crosswise setup: the board is rotated 180 degrees for Black (sq -> 63 - sq), which
    maps the starting position exactly onto itself;
  - facing setup: the board is mirrored top-to-bottom for Black (sq -> sq ^ 56).

Input planes (NUM_PLANES x 8 x 8), row = canonical rank, column = canonical file:
  0-5   own Padati, Ashva, Gaja, Ratha, Mantri, Raja
  6-11  opponent Padati, Ashva, Gaja, Ratha, Mantri, Raja
  12    1 if the current position has occurred before (repetition)
  13    no-progress counter / no_progress_plies (constant plane)
  14    ply / max_plies (constant plane)

Actions: from_sq * 64 + to_sq in canonical coordinates (ACTION_SIZE = 4096).
"""

import numpy as np

from .constants import WHITE
from .game import GameState

NUM_PLANES = 15
ACTION_SIZE = 64 * 64

_ROTATE = tuple(63 - sq for sq in range(64))
_MIRROR = tuple(sq ^ 56 for sq in range(64))
_IDENTITY = tuple(range(64))


def square_map(side: int, setup: str) -> tuple:
    """Absolute square -> canonical square for the side to move. The map is its own inverse."""
    if side == WHITE:
        return _IDENTITY
    return _ROTATE if setup == "crosswise" else _MIRROR


def move_to_action(move: int, side: int, setup: str) -> int:
    m = square_map(side, setup)
    fr, to = divmod(move, 64)
    return m[fr] * 64 + m[to]


def action_to_move(action: int, side: int, setup: str) -> int:
    # square_map is an involution, so decoding uses the same map.
    return move_to_action(action, side, setup)


def encode_state(state: GameState) -> np.ndarray:
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    m = square_map(state.side, state.rules.setup)
    s = state.side
    for sq, p in enumerate(state.board):
        if p:
            kind = p * s  # > 0 for own pieces
            plane = kind - 1 if kind > 0 else 6 + (-kind) - 1
            c = m[sq]
            planes[plane, c >> 3, c & 7] = 1.0
    if state.repetition_count() > 1:
        planes[12] = 1.0
    planes[13] = state.no_progress / state.rules.no_progress_plies
    planes[14] = state.ply / state.rules.max_plies
    return planes


def legal_actions(state: GameState) -> list:
    side, setup = state.side, state.rules.setup
    return [move_to_action(mv, side, setup) for mv in state.legal_moves()]


def legal_action_mask(state: GameState) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[legal_actions(state)] = True
    return mask
