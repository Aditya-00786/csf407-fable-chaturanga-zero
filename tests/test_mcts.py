"""Milestone 2: the search expands, selects and backs up values correctly.

Most tests use UniformEvaluator (equal priors, neutral value), so any tactical skill comes from
the search itself: it must find wins and avoid losses that the engine scores at game end.
"""

import numpy as np
import pytest

from chaturanga import BLACK, GameState, move_to_str, parse_move
from chaturanga.encoding import legal_action_mask
from chaturanga.mcts import MCTS, MCTSConfig, UniformEvaluator

MATE_IN_ONE = "k7/7p/1K6/8/8/8/8/6R1 w"             # Rg8#
BARE_KING_IN_ONE = "8/8/1k6/g7/8/8/8/R3K3 w"        # Rxa5 leaves Black a bare Raja
STALEMATE_TRAP = "7k/7p/4K2P/8/8/8/8/R7 w"          # Ra8# wins; Kf7 or Rg1 stalemate and lose


def colour_flipped(fen: str) -> str:
    """The same position with colours swapped, rotated 180 degrees (the crosswise setup's symmetry)."""
    board, side = fen.split()
    ranks = ["".join(ch.swapcase() for ch in reversed(rank)) for rank in reversed(board.split("/"))]
    return "/".join(ranks) + (" b" if side == "w" else " w")


def flipped_move(text: str) -> str:
    def sq(s: str) -> str:
        return "abcdefgh"[7 - "abcdefgh".index(s[0])] + str(9 - int(s[1]))
    return sq(text[:2]) + sq(text[2:])


def best_move(fen: str, simulations: int = 200, seed: int = 0) -> str:
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=simulations), seed=seed)
    root = mcts.search(GameState.from_fen(fen))
    return move_to_str(mcts.choose_move(root))


def test_colour_flip_helper():
    s = GameState.from_fen(colour_flipped(MATE_IN_ONE))
    assert s.side == BLACK and s.apply(parse_move(flipped_move("g1g8"))).outcome().winner == BLACK


@pytest.mark.parametrize("fen,answer", [
    (MATE_IN_ONE, "g1g8"),
    (BARE_KING_IN_ONE, "a1a5"),
    (STALEMATE_TRAP, "a1a8"),
])
@pytest.mark.parametrize("colour", ["white", "black"])
def test_finds_the_winning_move_for_either_side(fen, answer, colour):
    if colour == "black":
        fen, answer = colour_flipped(fen), flipped_move(answer)
    assert best_move(fen) == answer


def test_backed_up_values_have_the_right_sign():
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=800), seed=0)
    state = GameState.from_fen(STALEMATE_TRAP)
    root = mcts.search(state)
    # Child values are stored for the player who made the move. A move that ends the game holds
    # its exact result: +1 for the mate, -1 for a stalemate (the stalemated side wins).
    finished = 0
    for move, child in root.children.items():
        outcome = state.apply(move).outcome()
        if outcome is not None and child.visits:
            assert child.mean_value == pytest.approx(outcome.value_for(state.side))
            finished += 1
    assert finished >= 2
    assert MCTS.root_value(root) > 0.5
    assert root.children[parse_move("a1a8")].visits == max(MCTS.visit_counts(root).values())


class FixedValueEvaluator(UniformEvaluator):
    def __init__(self, value):
        self.value = value

    def evaluate(self, state):
        return super().evaluate(state)[0], self.value


def test_root_value_is_for_the_side_to_move():
    for fen in (MATE_IN_ONE, colour_flipped(MATE_IN_ONE)):
        mcts = MCTS(FixedValueEvaluator(0.4), MCTSConfig(simulations=0))
        assert MCTS.root_value(mcts.search(GameState.from_fen(fen))) == pytest.approx(0.4)
    # One level down every leaf is valued 0.4 for the opponent, so the root moves towards -0.4.
    mcts = MCTS(FixedValueEvaluator(0.4), MCTSConfig(simulations=10))
    assert MCTS.root_value(mcts.search(GameState.initial())) < 0


def test_visits_add_up_and_only_legal_moves_are_searched():
    state = GameState.initial()
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=120), seed=0)
    root = mcts.search(state)
    assert set(root.children) == set(state.legal_moves())
    assert sum(MCTS.visit_counts(root).values()) == 120
    assert root.visits == 121  # the expansion plus one per simulation


def test_every_expanded_node_holds_only_legal_children():
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=300), seed=3)
    stack = [mcts.search(GameState.initial())]
    seen = 0
    while stack:
        node = stack.pop()
        if node.state is None or not node.expanded:
            continue
        assert set(node.children) == set(node.state.legal_moves())
        assert sum(c.visits for c in node.children.values()) <= node.visits
        seen += 1
        stack.extend(node.children.values())
    assert seen > 10


def test_finished_games_cannot_be_searched():
    won = GameState.from_fen(MATE_IN_ONE).play("g1g8")
    with pytest.raises(ValueError):
        MCTS(UniformEvaluator()).search(won)


def test_tree_is_reused_for_the_next_move():
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=200), seed=0)
    state = GameState.initial()
    root = mcts.search(state)
    mine = mcts.choose_move(root)
    reply, node = max(root.children[mine].children.items(), key=lambda kv: kv[1].visits)
    before = node.visits
    assert before > 0
    later = mcts.search(state.apply(mine).apply(reply))
    assert later is node and later.visits == before + 200


def test_same_seed_same_search():
    def counts(seed):
        mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=150), seed=seed)
        return MCTS.visit_counts(mcts.search(GameState.initial(), add_noise=True))
    assert counts(7) == counts(7)
    assert counts(7) != counts(8)  # the root noise differs


def test_root_noise_keeps_a_distribution():
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=1), seed=1)
    root = mcts.search(GameState.initial(), add_noise=True)
    priors = [c.prior for c in root.children.values()]
    assert sum(priors) == pytest.approx(1.0)
    assert max(priors) > 1 / 16 + 1e-6  # no longer uniform


def test_policy_target_is_a_distribution_over_legal_actions():
    for fen in ("rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR w", colour_flipped(STALEMATE_TRAP)):
        state = GameState.from_fen(fen)
        pi = MCTS.policy_target(MCTS(UniformEvaluator(), MCTSConfig(simulations=64), seed=0).search(state))
        assert pi.sum() == pytest.approx(1.0)
        assert not pi[~legal_action_mask(state)].any()


def test_temperature_sampling_stays_legal():
    state = GameState.initial()
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=64), seed=0)
    root = mcts.search(state)
    picks = {mcts.choose_move(root, temperature=1.0) for _ in range(50)}
    assert picks <= set(state.legal_moves()) and len(picks) > 1
    assert mcts.choose_move(root, temperature=0.0) in state.legal_moves()


def test_self_play_game_with_the_search_finishes_legally():
    """A short noisy self-play game: the tree is reused move after move and every move is legal."""
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=24), seed=5)
    state = GameState.from_fen(STALEMATE_TRAP)
    for _ in range(40):
        if state.is_terminal():
            break
        move = mcts.choose_move(mcts.search(state, add_noise=True), temperature=1.0)
        assert state.is_legal(move)
        state = state.apply(move)
    assert np.isfinite(MCTS.root_value(mcts.root))
    assert state.is_terminal() or state.ply == 40
