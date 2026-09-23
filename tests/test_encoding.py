import numpy as np

from chaturanga import BLACK, GameState, RulesConfig, parse_square
from chaturanga.encoding import (
    ACTION_SIZE, NUM_PLANES, action_to_move, encode_state, legal_action_mask, move_to_action,
)


def test_shape_and_dtype():
    x = encode_state(GameState.initial())
    assert x.shape == (NUM_PLANES, 8, 8) and x.dtype == np.float32


def test_initial_piece_planes():
    x = encode_state(GameState.initial())
    assert x[0].sum() == 8 and x[6].sum() == 8        # Padatis
    assert x[5].sum() == 1 and x[5, 0, 4] == 1        # own Raja on e1
    assert x[11].sum() == 1 and x[11, 7, 3] == 1      # opponent Raja on d8
    assert x[12:].sum() == 0


def test_black_sees_the_same_start_position_as_white():
    # After rotation, Black's view of the crosswise start is identical to White's.
    white_view = encode_state(GameState.initial())
    fen = GameState.initial().to_fen().replace(" w ", " b ")
    black_view = encode_state(GameState.from_fen(fen))
    assert np.array_equal(white_view[:12], black_view[:12])


def test_black_sees_the_same_start_position_with_facing_setup():
    rules = RulesConfig(setup="facing")
    white_view = encode_state(GameState.initial(rules))
    fen = GameState.initial(rules).to_fen().replace(" w ", " b ")
    black_view = encode_state(GameState.from_fen(fen, rules))
    assert np.array_equal(white_view[:12], black_view[:12])


def test_action_round_trip_for_every_action():
    for setup in ("crosswise", "facing"):
        for side in (1, -1):
            for a in range(ACTION_SIZE):
                assert move_to_action(action_to_move(a, side, setup), side, setup) == a


def test_black_move_maps_to_the_mirrored_white_move():
    # Black's d7d6 under 180-degree rotation is e2e3 from Black's own point of view.
    d7d6 = parse_square("d7") * 64 + parse_square("d6")
    e2e3 = parse_square("e2") * 64 + parse_square("e3")
    assert move_to_action(d7d6, BLACK, "crosswise") == e2e3


def test_legal_mask_matches_legal_moves():
    s = GameState.initial().play("e2e3").play("d7d6").play("f1d3")
    mask = legal_action_mask(s)
    assert mask.shape == (ACTION_SIZE,)
    assert mask.sum() == len(s.legal_moves())
    decoded = {action_to_move(int(a), s.side, s.rules.setup) for a in np.flatnonzero(mask)}
    assert decoded == set(s.legal_moves())


def test_counter_planes():
    s = GameState.initial().play("g1h3").play("g8h6").play("h3g1").play("h6g8")
    x = encode_state(s)
    assert x[12].min() == 1.0                          # position seen before
    assert np.allclose(x[13], 4 / 140)
    assert np.allclose(x[14], 4 / 300)
