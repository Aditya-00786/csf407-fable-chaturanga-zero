"""Check and move legality (RULES.md section 4)."""

from conftest import destinations, position

from chaturanga import BLACK, move_to_str


ALL_KING_STEPS = {"c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5"}


def test_raja_cannot_step_into_attack_from_each_piece_type():
    # (black piece, its square, squares next to d4 it attacks)
    cases = [
        ("r", "h3", {"c3", "d3", "e3"}),   # Ratha covers rank 3
        ("g", "f5", {"d3"}),               # Gaja jumps f5 -> d3
        ("n", "b2", {"d3", "c4"}),         # Ashva
        ("m", "b4", {"c3", "c5"}),         # Mantri, one diagonal step
        ("p", "c6", {"d5"}),               # black Padati captures downwards: b5, d5
    ]
    for letter, sq, attacked in cases:
        s = position({"d4": "K", "h8": "k", sq: letter})
        assert destinations(s, "d4") == ALL_KING_STEPS - attacked, letter


def test_rajas_may_not_stand_next_to_each_other():
    s = position({"d4": "K", "d6": "k", "a8": "p"})
    assert not destinations(s, "d4") & {"c5", "d5", "e5"}


def test_pinned_piece_cannot_leave_the_pin_line():
    s = position({"e1": "K", "e2": "N", "e8": "r", "a8": "k"})
    assert destinations(s, "e2") == set()


def test_pinned_ratha_can_move_along_the_pin_line():
    s = position({"e1": "K", "e2": "R", "e8": "r", "a8": "k"})
    assert destinations(s, "e2") == {"e3", "e4", "e5", "e6", "e7", "e8"}


def test_must_get_out_of_check():
    s = position({"e1": "K", "e8": "r", "a3": "R", "a8": "k"})
    assert s.in_check()
    moves = {move_to_str(m) for m in s.legal_moves()}
    # Raja steps off the e-file, or the Ratha blocks on e3.
    assert moves == {"e1d1", "e1f1", "e1d2", "e1f2", "a3e3"}


def test_gaja_check_cannot_be_blocked():
    # Black Gaja on c3 attacks e1 by jumping over d2; a piece on d2 does not help.
    s = position({"e1": "K", "d2": "P", "c3": "g", "a8": "k", "b1": "N"})
    assert s.in_check()
    for m in s.legal_moves():
        fr, to = divmod(m, 64)
        assert move_to_str(m).startswith("e1") or move_to_str(m).endswith("c3")


def test_capturing_the_checker_is_legal():
    s = position({"e1": "K", "d2": "p", "a8": "k", "c1": "R"})  # black Padati d2 checks e1
    assert s.in_check()
    assert "c1d1" not in {move_to_str(m) for m in s.legal_moves()}
    assert "e1d2" in {move_to_str(m) for m in s.legal_moves()}


def test_moving_into_discovered_check_is_illegal():
    s = position({"e1": "K", "e4": "G", "e8": "r", "a8": "k"})
    assert destinations(s, "e4") == set()


def test_black_king_cannot_step_into_white_padati_attack():
    s = position({"e5": "k", "e3": "P", "a1": "K", "h1": "R"}, side=BLACK)
    assert "d4" not in destinations(s, "e5") and "f4" not in destinations(s, "e5")
