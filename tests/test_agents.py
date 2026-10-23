from conftest import position

from chaturanga import BLACK, GameState, move_to_str
from chaturanga.agents import GreedyAgent, RandomAgent, make_agent, material_balance


def test_random_agent_plays_legal_moves():
    agent, s = RandomAgent(seed=1), GameState.initial()
    for _ in range(40):
        if s.is_terminal():
            break
        move = agent.select_move(s)
        assert move in s.legal_moves()
        s = s.apply(move)


def test_material_balance_is_zero_at_start():
    assert material_balance(GameState.initial(), 1) == 0


def test_greedy_takes_a_free_ratha():
    s = position({"e1": "K", "d4": "N", "e6": "r", "h8": "k", "a7": "p"})
    assert move_to_str(GreedyAgent(seed=0).select_move(s)) == "d4e6"


def test_greedy_delivers_checkmate_in_one():
    s = position({"a8": "k", "h7": "p", "b6": "K", "g1": "R"})
    move = GreedyAgent(seed=0).select_move(s)
    assert s.apply(move).outcome().reason == "checkmate"


def test_greedy_does_not_hang_its_ratha_for_a_padati():
    # Taking the Padati on d5 with the Ratha loses the Ratha to the Padati on e6.
    s = position({"e1": "K", "d1": "R", "d5": "p", "e6": "p", "h8": "k"})
    assert move_to_str(GreedyAgent(seed=0).select_move(s)) != "d1d5"


def test_greedy_plays_legal_moves_for_black():
    s = GameState.initial().play("e2e3")
    assert s.side == BLACK
    assert GreedyAgent(seed=3).select_move(s) in s.legal_moves()


def test_make_agent():
    assert make_agent("greedy").id == "greedy"
    try:
        make_agent("alphazero")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown agents must be rejected")


def test_the_levels_that_play_their_best_move_follow_the_search_rules():
    from chaturanga.agents import LEVELS, MCTSAgent
    from chaturanga.mcts import UniformEvaluator

    assert {lv.id for lv in LEVELS.values() if lv.sure} == {"hard", "impossible"}
    assert all(lv.temperature == 0 for lv in LEVELS.values() if lv.sure)
    for sure in (False, True):
        config = MCTSAgent(evaluator=UniformEvaluator(), sure=sure).mcts.config
        assert config.take_wins == config.avoid_losses == sure
    assert not MCTSAgent(evaluator=UniformEvaluator()).mcts.config.take_wins  # Milestone 2's baseline as it was
