import pytest

from chaturanga import BLACK, WHITE, GameState, RulesConfig, move_to_str, parse_square
from chaturanga.constants import MANTRI, RAJA


def test_crosswise_setup_matches_rules_md():
    s = GameState.initial()
    assert s.to_fen() == "rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR w 0 0"
    assert s.board[parse_square("e1")] == RAJA
    assert s.board[parse_square("d1")] == MANTRI
    assert s.board[parse_square("d8")] == -RAJA
    assert s.board[parse_square("e8")] == -MANTRI
    assert s.side == WHITE


def test_facing_setup():
    s = GameState.initial(RulesConfig(setup="facing"))
    assert s.board[parse_square("e8")] == -RAJA
    assert s.board[parse_square("d8")] == -MANTRI


def test_crosswise_setup_is_symmetric_under_rotation():
    b = GameState.initial().board
    assert all(b[sq] == -b[63 - sq] for sq in range(64))


def test_facing_setup_is_symmetric_under_mirror():
    b = GameState.initial(RulesConfig(setup="facing")).board
    assert all(b[sq] == -b[sq ^ 56] for sq in range(64))


def test_start_position_has_16_moves():
    moves = sorted(move_to_str(m) for m in GameState.initial().legal_moves())
    assert moves == sorted([
        "a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "f2f3", "g2g3", "h2h3",  # Padati, single step only
        "b1a3", "b1c3", "g1f3", "g1h3",                                  # Ashva
        "c1a3", "c1e3", "f1d3", "f1h3",                                  # Gaja
    ])


def test_black_also_has_16_moves_at_start():
    s = GameState.initial().play("e2e3")
    assert s.side == BLACK
    assert len(s.legal_moves()) == 16


def test_fen_round_trip():
    s = GameState.initial().play("e2e3").play("d7d6").play("g1f3")
    assert GameState.from_fen(s.to_fen()).to_fen() == s.to_fen()


def test_fen_requires_one_raja_each():
    with pytest.raises(ValueError):
        GameState.from_fen("8/8/8/8/8/8/8/4K3 w")


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        RulesConfig(promotion="queen")
    with pytest.raises(ValueError):
        RulesConfig(setup="random")


def test_illegal_move_is_rejected():
    with pytest.raises(ValueError):
        GameState.initial().play("e2e4")  # no Padati double step
