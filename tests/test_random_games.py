"""Fuzz test: random games must stay consistent and always terminate."""

import random

from chaturanga import GameState, RulesConfig
from chaturanga.constants import PADATI, RAJA
from chaturanga.tables import KING_MOVES, is_attacked
from chaturanga.zobrist import compute_hash


def check_invariants(s):
    b = s.board
    assert b.count(RAJA) == 1 and b.count(-RAJA) == 1
    assert s.kings == (b.index(RAJA), b.index(-RAJA))
    assert s.kings[1] not in KING_MOVES[s.kings[0]]            # Rajas never touch
    assert s.material == (sum(0 < p < RAJA for p in b), sum(-RAJA < p < 0 for p in b))
    assert all(abs(b[sq]) != PADATI for sq in list(range(8)) + list(range(56, 64)))
    assert s.hash == compute_hash(b, s.side)
    assert s.history[-1] == s.hash
    # The side that just moved may not be in check.
    assert not is_attacked(b, s.king_square(-s.side), s.side)


def play_random_games(n, rules, seed):
    rng = random.Random(seed)
    reasons = {}
    for _ in range(n):
        s = GameState.initial(rules)
        while not s.is_terminal():
            check_invariants(s)
            moves = s.legal_moves()
            assert moves
            s = s.apply(rng.choice(moves))
        check_invariants(s)
        assert s.ply <= rules.max_plies
        reasons[s.outcome().reason] = reasons.get(s.outcome().reason, 0) + 1
    return reasons


def test_random_games_default_rules():
    reasons = play_random_games(300, RulesConfig(), seed=7)
    assert sum(reasons.values()) == 300


def test_random_games_alternative_rules():
    for rules in (
        RulesConfig(setup="facing"),
        RulesConfig(promotion="file"),
        RulesConfig(promotion="file_restricted"),
        RulesConfig(stalemate="draw", max_plies=150),
    ):
        play_random_games(40, rules, seed=11)
