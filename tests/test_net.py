"""Milestone 2: the untrained policy-value network and its use by the search and the MCTS agent."""

import pytest

torch = pytest.importorskip("torch")

from chaturanga import GameState, move_to_str  # noqa: E402
from chaturanga.agents import MCTSAgent, make_agent  # noqa: E402
from chaturanga.encoding import ACTION_SIZE, NUM_PLANES, encode_state  # noqa: E402
from chaturanga.mcts import MCTS, MCTSConfig  # noqa: E402
from chaturanga.net import (  # noqa: E402
    NetConfig, NetworkEvaluator, load_checkpoint, new_network, parameter_count, save_checkpoint,
)

SMALL = NetConfig(blocks=2, channels=16)


def test_output_shapes_and_value_range():
    net = new_network(SMALL)
    x = torch.randn(5, NUM_PLANES, 8, 8)
    logits, value = net(x)
    assert logits.shape == (5, ACTION_SIZE) and value.shape == (5,)
    assert torch.all(value.abs() <= 1)


def test_default_size_matches_the_proposal():
    net = new_network()
    assert net.config.blocks in range(4, 7) and net.config.channels in (64, 128)
    assert 500_000 < parameter_count(net) < 2_000_000


def test_same_seed_same_weights():
    a, b, c = new_network(SMALL, seed=1), new_network(SMALL, seed=1), new_network(SMALL, seed=2)
    x = torch.from_numpy(encode_state(GameState.initial())).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(a(x)[0], b(x)[0])
        assert not torch.equal(a(x)[0], c(x)[0])


def test_new_network_leaves_the_global_seed_alone():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    new_network(SMALL, seed=9)
    assert torch.equal(torch.rand(3), expected)


@pytest.mark.parametrize("fen", [
    "rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR w",
    "rngkmgnr/pppppppp/8/8/8/8/PPPPPPPP/RNGMKGNR b",
    "7k/7p/4K2P/8/8/8/8/R7 w",
])
def test_evaluator_priors_cover_exactly_the_legal_moves(fen):
    state = GameState.from_fen(fen)
    priors, value = NetworkEvaluator(new_network(SMALL), torch.device("cpu")).evaluate(state)
    assert set(priors) == set(state.legal_moves())
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(p > 0 for p in priors.values()) and -1 <= value <= 1


def test_batched_and_single_evaluation_agree():
    ev = NetworkEvaluator(new_network(SMALL), torch.device("cpu"))
    states = [GameState.initial(), GameState.initial().play("e2e3"), GameState.from_fen("7k/7p/4K2P/8/8/8/8/R7 w")]
    for single, batched in zip([ev.evaluate(s) for s in states], ev.evaluate_many(states)):
        assert single[1] == pytest.approx(batched[1], abs=1e-5)
        assert all(single[0][m] == pytest.approx(batched[0][m], abs=1e-5) for m in single[0])


def test_checkpoint_round_trip(tmp_path):
    net = new_network(SMALL, seed=4)
    path = tmp_path / "net.pt"
    save_checkpoint(str(path), net, games=0)
    loaded, extra = load_checkpoint(str(path))
    assert loaded.config == SMALL and extra == {"games": 0}
    x = torch.from_numpy(encode_state(GameState.initial())).unsqueeze(0)
    with torch.no_grad():
        assert torch.allclose(net(x)[0], loaded(x)[0]) and torch.allclose(net(x)[1], loaded(x)[1])


def test_search_with_the_untrained_network_still_finds_mate():
    """Priors and values are noise before training; the exact game-end values still win."""
    ev = NetworkEvaluator(new_network(SMALL), torch.device("cpu"))
    mcts = MCTS(ev, MCTSConfig(simulations=200), seed=0)
    root = mcts.search(GameState.from_fen("7k/7p/4K2P/8/8/8/8/R7 w"))
    assert move_to_str(mcts.choose_move(root)) == "a1a8"


def test_mcts_agent_plays_legal_moves():
    agent = make_agent("mcts", seed=0)
    assert isinstance(agent, MCTSAgent)
    state = GameState.initial()
    for _ in range(12):
        move = agent.select_move(state)
        assert state.is_legal(move)
        state = state.apply(move)
