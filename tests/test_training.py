"""Milestone 3: self-play records, the replay buffer, the training step and the training loop.

Self-play here uses short games (a 40-ply cap) and a few simulations, so the tests stay fast;
the logic is the same as in a full run.
"""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from chaturanga import GameState, move_to_str
from chaturanga.config import RulesConfig
from chaturanga.mcts import MCTS, MCTSConfig, UniformEvaluator
from chaturanga.net import NetConfig, NetworkEvaluator, load_checkpoint, new_network
from chaturanga.replay_buffer import ReplayBuffer
from chaturanga.selfplay import (GameRecord, SelfPlayConfig, dense_policy, examples, play_games, policy_target,
                                 read_games, write_games)
from chaturanga.training import TrainConfig, TrainingRun, held_out_loss, train_steps

SHORT = RulesConfig(max_plies=40)
FAST = SelfPlayConfig(simulations=8, temperature_plies=10, concurrent=3)
TINY = NetConfig(blocks=1, channels=16)


@pytest.fixture(scope="module")
def games():
    return play_games(UniformEvaluator(), 5, FAST, seed=3, rules=SHORT)


def test_self_play_records_are_legal_finished_games(games):
    assert len(games) == 5 and [g.seed for g in sorted(games, key=lambda g: g.seed)] == [3, 4, 5, 6, 7]
    for g in games:
        state = GameState.initial(SHORT)
        for text in g.moves:
            state = state.play(text)  # raises if illegal
        assert state.outcome().winner == g.winner and state.outcome().reason == g.reason
        assert len(g.visits) == g.plies and all(sum(v.values()) >= FAST.simulations for v in g.visits)


def test_same_seed_same_games(games):
    again = play_games(UniformEvaluator(), 5, FAST, seed=3, rules=SHORT)
    key = lambda gs: sorted((g.seed, g.moves) for g in gs)  # noqa: E731
    assert key(again) == key(games)


def test_batched_self_play_matches_one_game_at_a_time():
    together = play_games(UniformEvaluator(), 3, replace(FAST, concurrent=3), seed=11, rules=SHORT)
    alone = play_games(UniformEvaluator(), 3, replace(FAST, concurrent=1), seed=11, rules=SHORT)
    assert sorted(g.moves for g in together) == sorted(g.moves for g in alone)


def test_late_temperature_zero_plays_the_most_visited_move():
    (g,) = play_games(UniformEvaluator(), 1, replace(FAST, temperature_plies=4, late_temperature=0.0), seed=9, rules=SHORT)
    for ply, (move, visits) in enumerate(zip(g.moves, g.visits)):
        if ply >= 4:
            assert visits[move] == max(visits.values())


def test_examples_have_distribution_targets_and_signed_results(games):
    for g in games:
        planes, actions, shares, z = examples(g)
        assert planes.shape == (g.plies, 15, 8, 8) and len(actions) == len(shares) == len(z) == g.plies
        assert all(abs(s.sum() - 1) < 1e-5 for s in shares)
        if g.winner is None:
            assert not z.any()
        else:
            # Side to move alternates, so the result does too; the first position is White's.
            expected = np.array([g.winner * (1 if i % 2 == 0 else -1) for i in range(g.plies)], dtype=np.float32)
            assert (z == expected).all()


@pytest.mark.parametrize("fen", [None, "rngkmgnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNGMKGNR b"])
def test_policy_target_matches_the_search(fen):
    """Recorded visit counts give exactly MCTS.policy_target, for White and (rotated) for Black."""
    state = GameState.from_fen(fen) if fen else GameState.initial()
    mcts = MCTS(UniformEvaluator(), MCTSConfig(simulations=30), seed=0)
    root = mcts.search(state)
    visits = {move_to_str(m): c for m, c in MCTS.visit_counts(root).items() if c}
    assert np.allclose(dense_policy(*policy_target(state, visits)), MCTS.policy_target(root))


@pytest.mark.parametrize("tamper", ["illegal_move", "illegal_visit", "wrong_result", "unfinished"])
def test_tampered_records_are_rejected(games, tamper):
    g = GameRecord.from_json(json.loads(json.dumps(games[0].to_json())))
    if tamper == "illegal_move":
        g.moves[0] = "e2e5"
    elif tamper == "illegal_visit":
        g.visits[0]["a1a8"] = 3
    elif tamper == "wrong_result":
        g.winner, g.reason = (1, "checkmate") if g.winner != 1 else (-1, "checkmate")
    else:
        g.moves, g.visits = g.moves[:-1], g.visits[:-1]
    with pytest.raises(ValueError):
        examples(g)


def test_game_files_round_trip(games, tmp_path):
    path = tmp_path / "g.jsonl.gz"
    assert write_games(str(path), games) == len(games)
    back = list(read_games(str(path)))
    assert [b.to_json() for b in back] == [g.to_json() for g in games]


def test_buffer_keeps_the_newest_games_within_capacity(games):
    buf = ReplayBuffer(capacity=2 * max(g.plies for g in games))
    for g in games:
        buf.add(g)
    assert buf.games_added == 5 and len(buf) <= buf.capacity and buf.games >= 1
    kept = sum(g.plies for g in games[-buf.games:])
    assert len(buf) == kept  # exactly the newest games
    x, pi, z = buf.sample(64, np.random.default_rng(0))
    assert x.shape == (64, 15, 8, 8) and x.dtype == np.float32 and pi.shape == (64, 4096)
    assert np.allclose(pi.sum(1), 1) and set(np.unique(z)) <= {-1.0, 0.0, 1.0}


def test_training_fits_a_fixed_buffer(games):
    """The loss and gradients work: a small network can overfit a few games."""
    buf = ReplayBuffer()
    for g in games:
        buf.add(g)
    net = new_network(TINY, seed=0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = held_out_loss(net, games, torch.device("cpu"))
    train_steps(net, opt, buf, 150, 64, torch.device("cpu"), np.random.default_rng(0))
    last = held_out_loss(net, games, torch.device("cpu"))
    assert last["policy_loss"] < first["policy_loss"] - 0.5
    assert last["value_loss"] < first["value_loss"] * 0.5 or last["value_loss"] < 0.02
    assert not net.training  # left in evaluation mode for the search


def test_training_run_end_to_end_and_resume(tmp_path, games):
    cfg = TrainConfig(generations=1, games_per_generation=2, simulations=4, concurrent=2, steps_per_generation=5,
                      batch_size=16, eval_games=2, eval_opponents=("previous",), blocks=1, channels=16)
    run_dir = tmp_path / "runs" / "t"
    publish = tmp_path / "checkpoints" / "best.pt"
    quiet = lambda *_: None  # noqa: E731

    # A teammate's games wait in incoming/, next to a file with a tampered game.
    (run_dir / "incoming").mkdir(parents=True)
    write_games(str(run_dir / "incoming" / "aadit.jsonl.gz"),
                [replace(g, contributor="aadit", rules=asdict(RulesConfig())) for g in
                 play_games(UniformEvaluator(), 1, FAST, seed=50)])
    bad = replace(games[0], contributor="mallory")
    bad.moves = ["e2e5"] + bad.moves[1:]
    write_games(str(run_dir / "incoming" / "bad.jsonl.gz"), [bad])

    TrainingRun(str(run_dir), cfg, workers=0, device=torch.device("cpu"), publish=str(publish), log=quiet).run()
    events = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    kinds = {e["event"] for e in events}
    assert {"selfplay", "held_out", "train", "eval", "generation"} <= kinds
    selfplay = next(e for e in events if e["event"] == "selfplay")
    assert selfplay["games"] == 2 and selfplay["contributed"] == {"aadit": 1} and selfplay["rejected"] == 1
    assert not list((run_dir / "incoming").iterdir())
    assert [p.name for p in (run_dir / "games" / "rejected").iterdir()] == ["bad.jsonl.gz"]
    assert (run_dir / "checkpoints" / "gen_0001.pt").exists() and publish.exists()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "finished" and status["generation"] == 1

    # Resume with a higher target: generation 2 trains on from generation 1's weights and optimiser.
    TrainingRun(str(run_dir), replace(cfg, generations=2), workers=0, device=torch.device("cpu"),
                publish=str(publish), log=quiet).run()
    gens = [e["generation"] for e in map(json.loads, open(run_dir / "metrics.jsonl")) if e["event"] == "generation"]
    assert gens == [1, 2]
    _, meta = load_checkpoint(str(run_dir / "checkpoints" / "gen_0002.pt"))
    assert meta["generation"] == 2
    state = torch.load(run_dir / "checkpoints" / "train_state.pt", weights_only=True)
    assert state["step"] == 10


def test_training_steps_follow_the_replay_ratio(tmp_path):
    """Steps are sized so new positions are sampled about `sample_reuse` times, within the maximum."""
    run = TrainingRun(str(tmp_path / "r"), TrainConfig(batch_size=100, sample_reuse=4, steps_per_generation=300),
                      log=lambda *_: None)
    assert run.steps_for(1_000) == 40 and run.steps_for(50_000) == 300 and run.steps_for(1) == 1
    fixed = TrainingRun(str(tmp_path / "f"), TrainConfig(sample_reuse=0, steps_per_generation=300), log=lambda *_: None)
    assert fixed.steps_for(10) == 300


def test_network_size_of_a_run_cannot_change(tmp_path):
    cfg = TrainConfig(blocks=1, channels=16)
    TrainingRun(str(tmp_path / "r"), cfg, log=lambda *_: None)
    with pytest.raises(ValueError):
        TrainingRun(str(tmp_path / "r"), replace(cfg, channels=32), log=lambda *_: None)


def test_network_evaluator_drives_self_play():
    ev = NetworkEvaluator(new_network(TINY, seed=0), torch.device("cpu"))
    (g,) = play_games(ev, 1, FAST, seed=0, rules=SHORT)
    assert examples(g)[0].shape[0] == g.plies
