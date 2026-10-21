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


@pytest.fixture(scope="module")
def endgame_games():
    """Games from endgame starts with one piece a side: short, and the search sees their ends coming."""
    return play_games(UniformEvaluator(), 3, replace(FAST, simulations=24, endgame_fraction=1.0, endgame_max_pieces=1),
                      seed=5, rules=SHORT)


def test_records_keep_the_search_value_of_every_position(games, endgame_games):
    for g in games + endgame_games:
        assert len(g.values) == g.plies and all(-1.0 <= v <= 1.0 for v in g.values)
    # The uniform evaluator scores every unfinished position 0, so a value away from 0 is a result
    # the search saw coming, for the side to move: the winner, before the move that ends the game.
    decisive = [g for g in endgame_games if g.winner is not None]
    assert decisive and all(g.values[-1] > 0 for g in decisive)


def test_the_value_target_blends_the_result_with_the_search_value(endgame_games):
    g = max(endgame_games, key=lambda g: sum(abs(v) for v in g.values))
    z = examples(g)[3]
    plain, mixed = ReplayBuffer(), ReplayBuffer(value_mix=0.5)
    plain.add(g)
    mixed.add(g)
    assert np.allclose(plain._flatten()[3], z)
    assert np.allclose(mixed._flatten()[3], 0.5 * z + 0.5 * np.array(g.values), atol=1e-6)
    assert not np.allclose(mixed._flatten()[3], z)
    # A game recorded before search values were kept reads as before and trains on its result alone.
    data = g.to_json()
    data.pop("values")
    old = GameRecord.from_json(data)
    assert old.values == []
    fresh = ReplayBuffer(value_mix=0.5)
    fresh.add(old)
    assert np.allclose(fresh._flatten()[3], z)


def test_impossible_search_values_are_rejected(games):
    g = games[0]
    for values in (g.values[:-1], [2.0] + g.values[1:], [float("nan")] + g.values[1:]):
        with pytest.raises(ValueError):
            examples(replace(g, values=values))


def test_held_out_loss_scores_the_result_alone(games):
    """The honest measure stays how well the value predicts results, whatever training aims at."""
    net = new_network(TINY, seed=0)
    other = [replace(g, values=[0.9] * g.plies) for g in games]
    a, b = held_out_loss(net, games, torch.device("cpu")), held_out_loss(net, other, torch.device("cpu"))
    assert a["value_loss"] == pytest.approx(b["value_loss"])


def test_worker_processes_take_their_devices_in_turn(monkeypatch):
    """The first `gpu_workers` processes to start run the network on the GPU, the rest on their core."""
    import multiprocessing
    import signal

    from chaturanga import training

    started = multiprocessing.get_context("spawn").Value("i", 0)
    threads, handler = torch.get_num_threads(), signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(training, "_worker_device", training._worker_device)
    monkeypatch.setattr(training, "_progress", None)
    try:
        seen = []
        for _ in range(4):
            training._init_worker(None, None, ["cuda", "cpu", "cpu"], started)
            seen.append(training._worker_device.type)
    finally:
        torch.set_num_threads(threads)
        signal.signal(signal.SIGINT, handler)
    assert seen == ["cuda", "cpu", "cpu", "cuda"]  # a process restarted later takes the next turn


def test_a_resumed_run_trains_with_the_learning_rate_now_set(tmp_path):
    """The optimiser's saved state carries its old rate; the settings win when a run resumes."""
    cfg = TrainConfig(generations=1, games_per_generation=2, simulations=4, concurrent=2, steps_per_generation=3,
                      endgame_fraction=0, batch_size=16, eval_games=0, blocks=1, channels=16)
    TrainingRun(str(tmp_path / "r"), cfg, device=torch.device("cpu"), log=lambda *_: None).run()
    lower = replace(cfg, generations=2, learning_rate=1e-4, weight_decay=1e-3)
    run = TrainingRun(str(tmp_path / "r"), lower, device=torch.device("cpu"), log=lambda *_: None)
    run._load_or_create()
    assert {(g["lr"], g["weight_decay"]) for g in run.optimizer.param_groups} == {(1e-4, 1e-3)}
    starts = [json.loads(l) for l in open(tmp_path / "r" / "metrics.jsonl") if '"start"' in l]
    assert starts and starts[0]["after_generation"] == 0 and starts[0]["config"]["learning_rate"] == cfg.learning_rate


def test_gpu_workers_need_a_gpu(tmp_path):
    run = TrainingRun(str(tmp_path / "r"), TrainConfig(), workers=4, gpu_workers=6, log=lambda *_: None)
    assert run.gpu_workers == (4 if torch.cuda.is_available() else 0)


def test_a_run_trains_on_the_blend_it_is_set_to(tmp_path):
    run = TrainingRun(str(tmp_path / "r"), TrainConfig(value_mix=0.25), log=lambda *_: None)
    assert run.buffer.value_mix == 0.25


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
    cfg = TrainConfig(generations=1, games_per_generation=2, simulations=4, concurrent=2, steps_per_generation=5, endgame_fraction=0,
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


# ---------------------------------------------------------------------- pausing and resuming

from chaturanga.selfplay import read_games as _read  # noqa: E402
from chaturanga.training import pending_pause, request_pause  # noqa: E402

# Full games from the opening (no endgame starts), so pauses land part way through self-play.
PAUSE_CFG = TrainConfig(generations=1, games_per_generation=5, simulations=4, concurrent=1, steps_per_generation=30, endgame_fraction=0,
                        sample_reuse=0, batch_size=16, eval_games=2, eval_opponents=("previous",), blocks=1, channels=16)


class PausingRun(TrainingRun):
    """Asks itself to pause (or raises Ctrl+C) the first time its status matches `condition`."""

    def __init__(self, *args, when, condition, **kwargs):
        super().__init__(*args, **kwargs)
        self._when, self._condition, self._asked = when, condition, False

    def set_status(self, **changes):
        super().set_status(**changes)
        if not self._asked and self._condition(self.status):
            self._asked = True
            if self._when == "ctrl_c":
                raise KeyboardInterrupt
            request_pause(self.dir, self._when)


def _run(run_dir, cfg=PAUSE_CFG, workers=0, **pause):
    cls = PausingRun if pause else TrainingRun
    run = cls(str(run_dir), cfg, workers=workers, device=torch.device("cpu"), log=lambda *_: None, **pause)
    run.run()
    return json.loads((run_dir / "status.json").read_text())


def _events(run_dir, kind):
    return [e for e in map(json.loads, open(run_dir / "metrics.jsonl")) if e["event"] == kind]


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    d = tmp_path_factory.mktemp("whole") / "r"
    _run(d)
    return d


def test_pause_requests_round_trip(tmp_path):
    assert pending_pause(tmp_path) is None
    request_pause(tmp_path, "after_phase")
    assert pending_pause(tmp_path) == "after_phase"
    request_pause(tmp_path, None)
    assert pending_pause(tmp_path) is None
    with pytest.raises(ValueError):
        request_pause(tmp_path, "tomorrow")


def test_pause_now_keeps_finished_self_play_games(tmp_path, uninterrupted):
    d = tmp_path / "r"
    status = _run(d, when="now", condition=lambda s: s.get("state") == "selfplay" and s.get("games_done", 0) >= 2)
    assert status["state"] == "paused" and status["paused_in"] == "selfplay" and status["paused_after"] is None
    kept = [r for f in (d / "games").glob("gen_0001.partial.*.jsonl.gz") for r in _read(str(f))]
    assert len(kept) == 2 and not (d / "games" / "gen_0001.jsonl.gz").exists()
    assert not (d / "control.json").exists()  # the request is used up

    status = _run(d)  # carry on
    assert status["state"] == "finished"
    assert not list((d / "games").glob("*.partial.*"))
    # Only the missing games were played, and they are exactly the games of an uninterrupted run.
    whole = {r.seed: r.moves for r in _read(str(uninterrupted / "games" / "gen_0001.jsonl.gz"))}
    resumed = {r.seed: r.moves for r in _read(str(d / "games" / "gen_0001.jsonl.gz"))}
    assert resumed == whole


def test_pause_after_phase_finishes_self_play_first(tmp_path):
    d = tmp_path / "r"
    status = _run(d, when="after_phase", condition=lambda s: s.get("state") == "selfplay" and s.get("games_done", 0) >= 1)
    assert status["state"] == "paused" and status["paused_after"] == "selfplay"
    assert len(list(_read(str(d / "games" / "gen_0001.jsonl.gz")))) == 5
    assert not (d / "checkpoints" / "gen_0001.pt").exists()  # training had not started
    assert _run(d)["state"] == "finished"
    assert (d / "checkpoints" / "gen_0001.pt").exists()


def test_pause_now_during_training_redoes_it_on_resume(tmp_path):
    d = tmp_path / "r"
    status = _run(d, when="now", condition=lambda s: s.get("state") == "training" and s.get("step", 0) >= 10)
    assert status["state"] == "paused" and status["paused_in"] == "training"
    assert not (d / "checkpoints" / "gen_0001.pt").exists()
    assert _run(d)["state"] == "finished"
    assert torch.load(d / "checkpoints" / "train_state.pt", weights_only=True)["step"] == 30  # not 40


def test_pause_now_during_evaluation_evaluates_on_resume(tmp_path):
    d = tmp_path / "r"
    status = _run(d, when="now", condition=lambda s: s.get("state") == "evaluating")
    assert status["state"] == "paused" and status["paused_in"] == "evaluating"
    assert (d / "checkpoints" / "gen_0001.pt").exists() and not _events(d, "generation")
    assert _run(d)["state"] == "finished"
    assert [e["generation"] for e in _events(d, "generation")] == [1]
    assert [e["generation"] for e in _events(d, "eval")] == [1]
    assert torch.load(d / "checkpoints" / "train_state.pt", weights_only=True)["step"] == 30  # not trained again


def test_pause_after_generation_and_carry_on(tmp_path):
    d = tmp_path / "r"
    cfg = replace(PAUSE_CFG, generations=2)
    status = _run(d, cfg, when="after_generation",
                  condition=lambda s: s.get("state") == "selfplay" and s.get("generation") == 1)
    assert status["state"] == "paused" and status["paused_after"] == "generation" and status["generation"] == 1
    assert [e["generation"] for e in _events(d, "generation")] == [1]
    assert _run(d, cfg)["state"] == "finished"
    assert [e["generation"] for e in _events(d, "generation")] == [1, 2]


def test_ctrl_c_pauses_like_now(tmp_path):
    d = tmp_path / "r"
    status = _run(d, when="ctrl_c", condition=lambda s: s.get("state") == "selfplay" and s.get("games_done", 0) >= 1)
    assert status["state"] == "paused"
    assert _run(d)["state"] == "finished"


def test_pause_now_stops_worker_processes_and_keeps_their_games(tmp_path):
    d = tmp_path / "r"
    # Enough games that the pause lands well before they are all played, however fast the machine.
    cfg = replace(PAUSE_CFG, games_per_generation=12)
    status = _run(d, cfg, workers=2, when="now",
                  condition=lambda s: s.get("state") == "selfplay" and s.get("games_done", 0) >= 1)
    assert status["state"] == "paused"
    kept = {r.seed for f in (d / "games").glob("gen_0001.partial.*.jsonl.gz") for r in _read(str(f))}
    assert 1 <= len(kept) < 12
    assert _run(d, cfg, workers=2)["state"] == "finished"
    assert len({r.seed for r in _read(str(d / "games" / "gen_0001.jsonl.gz"))}) == 12


def test_command_line_pause(tmp_path):
    import subprocess
    import sys

    TrainingRun(str(tmp_path / "r"), PAUSE_CFG, log=lambda *_: None)
    script = __import__("pathlib").Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    out = subprocess.run([sys.executable, str(script), "--runs-dir", str(tmp_path), "--run", "r", "--pause", "after-generation"],
                         capture_output=True, text=True, check=True).stdout
    assert "pause after generation" in out and pending_pause(tmp_path / "r") == "after_generation"
    subprocess.run([sys.executable, str(script), "--runs-dir", str(tmp_path), "--run", "r", "--cancel-pause"], check=True,
                   capture_output=True)
    assert pending_pause(tmp_path / "r") is None


# ---------------------------------------------------------------------- endgame starts

import random as _random  # noqa: E402

from chaturanga.endgames import random_endgame  # noqa: E402
from chaturanga.selfplay import start_position, summarise  # noqa: E402
from chaturanga.tables import is_attacked  # noqa: E402


def test_endgame_starts_are_legal_unfinished_positions():
    rng = _random.Random(0)
    for _ in range(300):
        s = random_endgame(rng, max_pieces=5)
        assert s.outcome() is None and s.legal_moves()
        for colour in (1, -1):
            pieces = [p for p in s.board if p * colour > 0]
            assert pieces.count(6 * colour) == 1 and 2 <= len(pieces) <= 6  # the Raja and 1 to 5 more
        assert not any(abs(p) == 1 and not 8 <= sq < 56 for sq, p in enumerate(s.board))  # Padatis off ranks 1 and 8
        assert not is_attacked(s.board, s.king_square(-s.side), s.side)  # the side not to move is not in check


def test_endgame_starts_follow_the_seed_and_the_fraction():
    cfg = SelfPlayConfig(endgame_fraction=0.3)
    starts = [start_position(seed, cfg) for seed in range(1000)]
    assert 0.25 < sum(s is not None for s in starts) / 1000 < 0.35
    assert [s and s.to_fen() for s in starts[:50]] == [s and s.to_fen() for s in (start_position(i, cfg) for i in range(50))]
    assert start_position(1, SelfPlayConfig()) is None  # off unless asked for


def test_endgame_games_record_and_replay_their_start():
    cfg = replace(FAST, endgame_fraction=1.0, endgame_max_pieces=2)
    games = play_games(UniformEvaluator(), 4, cfg, seed=5)
    for g in games:
        assert g.start_fen and examples(g)[0].shape[0] == g.plies
        moved = GameRecord.from_json(json.loads(json.dumps(g.to_json())))
        moved.start_fen = None  # the same moves from the standard setup are not this game
        with pytest.raises(ValueError):
            examples(moved)
    summary = summarise(games)
    assert summary["games"] == 0 and summary["endgames"]["games"] == 4 and summary["positions"] == sum(g.plies for g in games)


def test_a_run_started_with_auto_report_reports_as_it_finishes(tmp_path):
    from chaturanga.analysis import read_report, read_strength

    d = tmp_path / "r"
    cfg = replace(PAUSE_CFG, games_per_generation=2, auto_report=True, report_games=2)
    assert _run(d, cfg)["state"] == "finished"
    report = read_report(d)
    assert report and report["generations"] == 1 and report["probe"]["networks"]
    assert read_strength(d)["state"] == "done"


def test_without_auto_report_a_finished_run_has_no_report(tmp_path):
    from chaturanga.analysis import read_report

    d = tmp_path / "r"
    assert _run(d, replace(PAUSE_CFG, games_per_generation=2))["state"] == "finished"
    assert read_report(d) is None and not (d / "report").exists()


def test_progress_is_reported_while_games_are_played(tmp_path):
    """Live moves and finished games reach the status between finished batches, in every phase."""
    seen = []

    class Watching(TrainingRun):
        def set_status(self, **changes):
            super().set_status(**changes)
            if self.status.get("state") in ("selfplay", "evaluating") and self.status.get("moves"):
                seen.append((self.status["state"], self.status["moves"], self.status.get("games_done"),
                             self.status.get("moves_expected"), self.status.get("phase_started")))

    cfg = replace(PAUSE_CFG, games_per_generation=4)
    for workers in (0, 2):
        seen.clear()
        Watching(str(tmp_path / f"w{workers}"), cfg, workers=workers, device=torch.device("cpu"), log=lambda *_: None).run()
        for state in ("selfplay", "evaluating"):
            mine = [s for s in seen if s[0] == state]
            assert mine, f"no live progress during {state} with {workers} workers"
            moves = [s[1] for s in mine]
            assert moves == sorted(moves) and moves[-1] > 0  # it only grows within a phase
            assert all(s[3] and s[4] for s in mine)          # with an estimate and a start time
