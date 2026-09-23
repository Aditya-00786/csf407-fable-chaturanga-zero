import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # lets tests import reference.py

from chaturanga import WHITE, GameState, parse_square, square_name  # noqa: E402


def position(pieces, side=WHITE, **kwargs):
    return GameState.from_pieces(pieces, side, **kwargs)


def destinations(state, from_name):
    """Names of the squares the piece on `from_name` can legally move to."""
    fr = parse_square(from_name)
    return {square_name(m % 64) for m in state.legal_moves() if m // 64 == fr}
