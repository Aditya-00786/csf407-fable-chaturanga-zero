"""How a game ends (RULES.md section 5)."""

from conftest import position

from chaturanga import BLACK, WHITE, GameState, RulesConfig


def test_start_position_is_not_terminal():
    assert GameState.initial().outcome() is None


def test_checkmate():
    # Black Raja a8 checked along rank 8 by the Ratha; the white Raja on b6 covers a7 and b7.
    s = position({"a8": "k", "a3": "p", "h8": "R", "b6": "K"}, side=BLACK)
    assert s.in_check() and not s.legal_moves()
    out = s.outcome()
    assert (out.winner, out.reason) == (WHITE, "checkmate")


def test_stalemated_player_wins():
    # Black to move: Raja h8 boxed in by the white Raja f7 and Padati h6; Padati h7 is blocked.
    s = position({"h8": "k", "h7": "p", "h6": "P", "f7": "K"}, side=BLACK)
    assert not s.in_check() and not s.legal_moves()
    out = s.outcome()
    assert (out.winner, out.reason) == (BLACK, "stalemate")


def test_stalemate_alternatives_are_configurable():
    pieces = {"h8": "k", "h7": "p", "h6": "P", "f7": "K"}
    assert position(pieces, BLACK, rules=RulesConfig(stalemate="stalemating_wins")).outcome().winner == WHITE
    assert position(pieces, BLACK, rules=RulesConfig(stalemate="draw")).outcome().winner is None


def test_baring_the_king_wins_immediately():
    s = position({"e1": "K", "a1": "R", "b6": "k", "a5": "p"}).play("a1a5")
    out = s.outcome()
    # Black could recapture with Kb6xa5 and bare White too, but the game is already over.
    assert (out.winner, out.reason) == (WHITE, "bare_king")
    assert s.legal_moves()  # black would still have had moves


def test_bare_king_takes_priority_over_stalemate():
    # White's capture on e4 leaves Black with a bare Raja on a8 that also has no legal move:
    # b8 and b7 are covered by the Raja on c7, a7 by the Padati on b6.
    s = position({"a8": "k", "e4": "g", "c7": "K", "b6": "P", "e1": "R"})
    s = s.play("e1e4")
    assert not s.legal_moves() and not s.in_check()
    out = s.outcome()
    assert (out.winner, out.reason) == (WHITE, "bare_king")


def test_both_bare_is_a_draw():
    s = position({"e1": "K", "e8": "k"})
    assert s.outcome().winner is None


def test_threefold_repetition_is_a_draw():
    s = GameState.initial()
    shuffle = ["g1h3", "g8h6", "h3g1", "h6g8"]
    for mv in shuffle:
        s = s.play(mv)
    assert s.repetition_count() == 2 and s.outcome() is None
    for mv in shuffle:
        s = s.play(mv)
    assert s.repetition_count() == 3
    out = s.outcome()
    assert (out.winner, out.reason) == (None, "repetition")


def test_padati_move_resets_repetition_history():
    s = GameState.initial().play("g1h3").play("g8h6").play("h3g1").play("h6g8")
    s = s.play("a2a3")
    assert s.history == (s.hash,)
    assert s.no_progress == 0


def test_no_progress_draw_after_140_plies():
    pieces = {"e1": "K", "a1": "R", "e8": "k", "h8": "r"}
    s = position(pieces, no_progress=139).play("a1a2")
    out = s.outcome()
    assert s.no_progress == 140
    assert (out.winner, out.reason) == (None, "no_progress")


def test_capture_resets_no_progress_counter():
    s = position({"e1": "K", "a1": "R", "a5": "p", "a6": "p", "e8": "k"}, no_progress=100).play("a1a5")
    assert s.no_progress == 0 and s.outcome() is None


def test_ply_cap():
    rules = RulesConfig(max_plies=4)
    s = GameState.initial(rules)
    for mv in ["e2e3", "e7e6", "d2d3", "d7d6"]:
        assert s.outcome() is None
        s = s.play(mv)
    out = s.outcome()
    assert (out.winner, out.reason) == (None, "max_plies")


def test_outcome_values():
    s = position({"a8": "k", "a3": "p", "h8": "R", "b6": "K"}, side=BLACK)
    out = s.outcome()
    assert out.value_for(WHITE) == 1 and out.value_for(BLACK) == -1
