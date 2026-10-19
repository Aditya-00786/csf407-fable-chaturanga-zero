"""Random reduced-material starting positions for self-play ("endgame starts").

From the opening, games run about 250 plies and fewer than half are decided, so the value head
sees few, noisy results. A game started with only a few pieces each is short and usually
decisive, and when the material is uneven it shows which pieces win games. The positions use
nothing but the rules: each side gets its Raja and a random handful of its own original pieces,
placed at random, and the position must be legal and not already finished. No evaluation is
written in.
"""

import random

from .config import DEFAULT_RULES, RulesConfig
from .constants import BLACK, INITIAL_COUNTS, PADATI, PIECE_LETTERS, RAJA, WHITE, square_name
from .game import GameState
from .tables import is_attacked

# Padatis never stand on the first or last rank: they start on the second and promote on the last.
_PADATI_SQUARES = tuple(range(8, 56))


def random_endgame(rng: random.Random, max_pieces: int = 5, rules: RulesConfig = DEFAULT_RULES) -> GameState:
    """A legal, unfinished position: each Raja plus 1 to `max_pieces` of that side's pieces."""
    pool = [kind for kind, n in INITIAL_COUNTS.items() for _ in range(n)]
    while True:
        board: dict[int, int] = {}
        for colour in (WHITE, BLACK):
            kinds = [RAJA] + rng.sample(pool, rng.randint(1, max_pieces))
            for kind in kinds:
                free = [sq for sq in (_PADATI_SQUARES if kind == PADATI else range(64)) if sq not in board]
                board[rng.choice(free)] = colour * kind
        side = rng.choice((WHITE, BLACK))
        squares = [0] * 64
        for sq, piece in board.items():
            squares[sq] = piece
        their_king = next(sq for sq, p in board.items() if p == -side * RAJA)
        if is_attacked(squares, their_king, side):
            continue  # the side not to move would be in check: not a legal position
        state = GameState.from_pieces(
            {square_name(sq): PIECE_LETTERS[abs(p)] if p > 0 else PIECE_LETTERS[abs(p)].lower() for sq, p in board.items()},
            side, rules)
        if state.outcome() is None:
            return state
