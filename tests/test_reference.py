"""Cross-check the engine against the independent reference generator (tests/reference.py)."""

import random

import reference

from chaturanga import GameState


def engine_perft(state, depth):
    if depth == 0:
        return 1
    moves = state.legal_moves()
    if depth == 1:
        return len(moves)
    return sum(engine_perft(state.apply(m), depth - 1) for m in moves)


def test_perft_start_position():
    s = GameState.initial()
    assert engine_perft(s, 1) == 16
    assert engine_perft(s, 2) == 256  # no first move can affect Black's replies
    assert engine_perft(s, 3) == reference.perft(s, 3)


def test_legal_moves_match_reference_along_random_games():
    rng = random.Random(1234)
    positions = 0
    for _ in range(60):
        s = GameState.initial()
        while not s.is_terminal():
            assert set(s.legal_moves()) == reference.legal_moves(s.board, s.side), s.to_fen()
            assert s.in_check() == reference.in_check(s.board, s.side)
            positions += 1
            s = s.apply(rng.choice(s.legal_moves()))
    assert positions > 5000
