"""Zobrist hashing of (board, side to move), used for repetition detection."""

import random

from .constants import WHITE

_rng = random.Random(0x43484154)  # fixed seed: hashes are identical across runs

# PIECE_KEYS[piece + 6][sq] for piece in -6..6 (index 6, the empty cell, is unused).
PIECE_KEYS = tuple(tuple(_rng.getrandbits(64) for _ in range(64)) for _ in range(13))
SIDE_KEY = _rng.getrandbits(64)  # XORed in when Black is to move


def compute_hash(board, side: int) -> int:
    h = 0 if side == WHITE else SIDE_KEY
    for sq, p in enumerate(board):
        if p:
            h ^= PIECE_KEYS[p + 6][sq]
    return h
