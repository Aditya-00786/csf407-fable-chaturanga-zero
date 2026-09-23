"""Piece movement rules (RULES.md section 3)."""

from conftest import destinations, position

from chaturanga import BLACK, GameState, RulesConfig, parse_square
from chaturanga.constants import ASHVA, GAJA, MANTRI, RATHA
from chaturanga.tables import GAJA_MOVES

# Kings parked in corners, away from the action.
KINGS = {"a1": "K", "h8": "k"}


def test_raja_steps_one_square_any_direction():
    s = position({"d4": "K", "h8": "k", "a8": "p"})
    assert destinations(s, "d4") == {"c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5"}


def test_no_castling():
    s = position({"e1": "K", "a1": "R", "h1": "R", "d8": "k", "a8": "p"})
    assert destinations(s, "e1") == {"d1", "f1", "d2", "e2", "f2"}


def test_mantri_steps_one_square_diagonally():
    s = position({**KINGS, "d4": "M"})
    assert destinations(s, "d4") == {"c3", "e3", "c5", "e5"}


def test_ratha_slides_orthogonally_and_is_blocked():
    s = position({**KINGS, "d4": "R", "d6": "P", "f4": "p"})
    assert destinations(s, "d4") == {
        "d5",                    # stops before own piece on d6
        "d3", "d2", "d1",
        "c4", "b4", "a4",
        "e4", "f4",              # captures the enemy on f4, no further
    }


def test_gaja_jumps_two_squares_diagonally():
    s = position({**KINGS, "d4": "G"})
    assert destinations(s, "d4") == {"b2", "f2", "b6", "f6"}


def test_gaja_jumps_over_pieces_and_captures():
    s = position({**KINGS, "d4": "G", "c3": "P", "e5": "p", "c5": "P", "e3": "p", "f6": "p", "b6": "P"})
    assert destinations(s, "d4") == {"b2", "f2", "f6"}  # b6 is own piece; f6 is a capture


def test_each_gaja_can_only_ever_reach_8_squares():
    for start in ("c1", "f1"):
        seen, frontier = set(), [parse_square(start)]
        while frontier:
            sq = frontier.pop()
            if sq not in seen:
                seen.add(sq)
                frontier.extend(GAJA_MOVES[sq])
        assert len(seen) == 8


def test_ashva_moves_like_a_knight_and_jumps():
    s = position({**KINGS, "d4": "N", "d5": "P", "e4": "P", "d3": "p", "c4": "p"})
    assert destinations(s, "d4") == {"b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"}


def test_padati_single_step_only():
    s = GameState.initial()
    assert destinations(s, "e2") == {"e3"}


def test_padati_blocked_and_diagonal_captures():
    s = position({**KINGS, "e4": "P", "e5": "p", "d5": "p", "f5": "P"})
    assert destinations(s, "e4") == {"d5"}  # blocked ahead, own piece on f5


def test_padati_never_moves_backwards_or_sideways():
    s = position({**KINGS, "e4": "P", "d3": "p", "e3": "p", "d4": "p"})
    assert destinations(s, "e4") == {"e5"}


def test_black_padati_moves_down():
    s = position({**KINGS, "e5": "p", "d4": "P"}, side=BLACK)
    assert destinations(s, "e5") == {"e4", "d4"}


def test_promotion_to_mantri():
    s = position({**KINGS, "c7": "P"}).play("c7c8")
    assert s.board[parse_square("c8")] == MANTRI


def test_capture_promotion_for_black():
    s = position({**KINGS, "b2": "p", "c1": "N"}, side=BLACK).play("b2c1")
    assert s.board[parse_square("c1")] == -MANTRI


def test_multiple_mantris_allowed():
    s = position({**KINGS, "d1": "M", "e7": "P"}).play("e7e8")
    assert s.board.count(MANTRI) == 2


def test_file_promotion():
    rules = RulesConfig(promotion="file")
    expected = {"a": RATHA, "b": ASHVA, "c": GAJA, "d": MANTRI, "e": MANTRI, "f": GAJA, "g": ASHVA}
    for f, kind in expected.items():
        s = position({"a1": "K", "h1": "k", f + "7": "P"}, rules=rules).play(f"{f}7{f}8")
        assert s.board[parse_square(f + "8")] == kind, f


def test_file_restricted_promotion_needs_a_lost_piece():
    rules = RulesConfig(promotion="file_restricted")
    both_rathas = position({"e1": "K", "h8": "k", "a1": "R", "h1": "R", "a7": "P"}, rules=rules)
    assert destinations(both_rathas, "a7") == set()  # may not step onto the last rank
    one_ratha = position({"e1": "K", "h8": "k", "a1": "R", "a7": "P"}, rules=rules)
    assert destinations(one_ratha, "a7") == {"a8"}
    assert one_ratha.play("a7a8").board[parse_square("a8")] == RATHA
