"""ChaturangZero: Chaturanga rules engine (rules: docs/RULES.md)."""

from .config import DEFAULT_RULES, RulesConfig
from .constants import (
    ASHVA, BLACK, EMPTY, GAJA, MANTRI, PADATI, RAJA, RATHA, WHITE,
    parse_square, square_name,
)
from .game import GameState, Outcome, decode_move, encode_move, move_to_str, parse_move

__all__ = [
    "DEFAULT_RULES", "RulesConfig", "GameState", "Outcome",
    "WHITE", "BLACK", "EMPTY", "PADATI", "ASHVA", "GAJA", "RATHA", "MANTRI", "RAJA",
    "encode_move", "decode_move", "move_to_str", "parse_move", "parse_square", "square_name",
]
