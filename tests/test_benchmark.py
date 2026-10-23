"""The rating ladder and the conversion test (chaturanga/benchmark.py)."""

import json
import random

import pytest

from chaturanga import WHITE, GameState
from chaturanga import benchmark
from chaturanga.benchmark import (RECIPES, elo_steps, fit_elo, pair_key, plan, read_benchmark, run_benchmark,
                                  won_positions)
from chaturanga.constants import LETTER_PIECES, RAJA
from chaturanga.mcts import wins_at_once
from chaturanga.net import NetConfig, new_network, save_checkpoint
from chaturanga.training import TrainConfig, play_matches


def _results(true: dict, games: int, seed: int = 1) -> dict:
    """Pair results drawn from Elo ratings (no draws)."""
    rng, names, pairs = random.Random(seed), list(true), {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = 1 / (1 + 10 ** ((true[b] - true[a]) / 400))
            w = sum(rng.random() < p for _ in range(games))
            pairs[pair_key(a, b)] = {"a": a, "b": b, "games": games, "a_wins": w, "draws": 0, "b_wins": games - w}
    return pairs


def test_the_fit_finds_the_ratings_that_made_the_results():
    true = {"random": 0, "greedy": 300, "gen_0000": 100, "gen_0010": 450, "gen_0020": 550}
    ratings = fit_elo(list(true), _results(true, 2000))
    assert ratings["random"]["elo"] == 0 and ratings["random"]["low"] == ratings["random"]["high"] == 0
    for name, elo in true.items():
        r = ratings[name]
        assert abs(r["elo"] - elo) < 30 and r["low"] <= r["elo"] <= r["high"]
        assert r["games"] == 4 * 2000
    # Fewer games, wider intervals.
    few = fit_elo(list(true), _results(true, 50))
    assert few["gen_0020"]["high"] - few["gen_0020"]["low"] > 3 * (ratings["gen_0020"]["high"] - ratings["gen_0020"]["low"])


def test_each_step_between_generations_has_its_own_narrower_interval():
    true = {"random": 0, "greedy": 300, "gen_0000": 100, "gen_0010": 450, "gen_0020": 550}
    pairs = _results(true, 200)
    ratings, steps = fit_elo(list(true), pairs), elo_steps(list(true), pairs)
    assert [(s["from"], s["to"]) for s in steps] == [(0, 10), (10, 20)]
    for s in steps:
        a, b = ratings[f"gen_{s['from']:04d}"], ratings[f"gen_{s['to']:04d}"]
        assert s["diff"] == pytest.approx(b["elo"] - a["elo"], abs=0.2) and s["low"] < s["diff"] < s["high"]
        # Narrower than treating the two ratings' intervals as independent.
        assert s["high"] - s["low"] < ((b["high"] - b["low"]) ** 2 + (a["high"] - a["low"]) ** 2) ** 0.5


def test_a_sweep_gives_a_finite_gap_and_unlinked_entries_no_rating():
    pairs = {pair_key("random", "gen_0001"): {"a": "random", "b": "gen_0001", "games": 40, "a_wins": 0, "draws": 0,
                                              "b_wins": 40}}
    ratings = fit_elo(["random", "greedy", "gen_0001"], pairs)
    assert 300 < ratings["gen_0001"]["elo"] < 1500  # one virtual draw: large, not endless
    assert ratings["greedy"] is None  # no games yet


def test_the_conversion_positions_are_fixed_and_one_side_is_far_ahead():
    positions = won_positions()
    assert positions == won_positions() and len(positions) == 8 * len(RECIPES)
    for i, p in enumerate(positions):
        state = GameState.from_fen(p["fen"])
        _, ahead, behind = next(r for r in RECIPES if r[0] == p["recipe"])
        mine = sorted(abs(x) for x in state.board if x and (x > 0) == (state.side > 0))
        theirs = sorted(abs(x) for x in state.board if x and (x > 0) != (state.side > 0))
        assert mine == sorted([RAJA] + [LETTER_PIECES[c] for c in ahead])
        assert theirs == sorted([RAJA] + [LETTER_PIECES[c] for c in behind])
        assert state.outcome() is None and not any(wins_at_once(state, m) for m in state.legal_moves())
        assert state.side == (WHITE if i % 8 % 2 == 0 else -WHITE)


def test_a_game_from_a_set_position_is_played_by_the_side_to_move():
    mate = "7k/7p/4K2P/8/8/8/8/R7 w"  # Greedy mates at once (Ra8)
    for fen, colour in ((mate, WHITE), ("r7/8/8/8/8/4k2p/7P/7K b", -WHITE)):
        (row,) = play_matches([("greedy", "random", 0, 5, fen)], simulations=2)
        assert row["colour"] == colour and row["score"] == 1 and row["reason"] == "checkmate" and row["plies"] == 1
    # From the opening, colours still alternate with the game's index, whoever plays.
    rows = play_matches([("random", "greedy", i, 5) for i in range(2)], simulations=2)
    assert [r["colour"] for r in rows] == [WHITE, -WHITE]


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    """A run folder with two small networks, and a conversion test cut to three positions."""
    run = tmp_path / "tiny"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(TrainConfig(simulations=2, eval_opening_plies=2).to_json()))
    for g in (0, 1):
        save_checkpoint(str(run / "checkpoints" / f"gen_{g:04d}.pt"), new_network(NetConfig(blocks=1, channels=8), seed=g))
    whole = won_positions

    monkeypatch.setattr(benchmark, "won_positions", lambda: whole(per_recipe=1)[:3])
    return run


def test_the_ladder_rates_every_entry_and_later_plays_only_what_is_new(tiny_run):
    data = run_benchmark(tiny_run, games=2, every=1, workers=1, log=lambda *_: None)
    entries = ["random", "greedy", "gen_0000", "gen_0001"]
    assert data["state"] == "done" and data["entries"] == entries
    assert len(data["pairs"]) == 6 and all(p["games"] == 2 for p in data["pairs"].values())
    assert data["done"] == data["total"] == 6 * 2 + 3 * 3  # the pairs, and three networks (Greedy too) converting
    assert all(data["ratings"][e] is not None for e in entries) and data["ratings"]["random"]["elo"] == 0
    assert set(data["conversion"]) == {"greedy", "gen_0000", "gen_0001"}
    assert all(c["games"] == 3 and 0 <= c["won"] <= 3 for c in data["conversion"].values())
    assert read_benchmark(tiny_run) == {**data, "updated": read_benchmark(tiny_run)["updated"]}

    # A new generation plays the others and converts; nothing else is played again.
    save_checkpoint(str(tiny_run / "checkpoints" / "gen_0002.pt"), new_network(NetConfig(blocks=1, channels=8), seed=2))
    todo = plan(tiny_run, games=2, every=1)
    assert todo["new"] == ["gen_0002"] and todo["ladder_games"] == 4 * 2 and todo["converting"] == ["gen_0002"]
    again = run_benchmark(tiny_run, games=2, every=1, workers=1, log=lambda *_: None)
    assert again["total"] == 4 * 2 + 3 and len(again["pairs"]) == 10
    assert {k: v for k, v in again["pairs"].items() if k in data["pairs"]} == data["pairs"]
    assert again["conversion"]["gen_0000"] == data["conversion"]["gen_0000"]
    # More games a pair adds to every pair.
    assert plan(tiny_run, games=3, every=1)["ladder_games"] == 10


def test_every_nth_generation_with_the_first_latest_and_best(tmp_path):
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    for g in range(0, 26):
        (run / "checkpoints" / f"gen_{g:04d}.pt").write_bytes(b"")
    import torch

    torch.save({"best": 17}, run / "checkpoints" / "train_state.pt")
    entries = plan(run, games=2, every=10)["entries"]
    assert entries == ["random", "greedy", "gen_0000", "gen_0010", "gen_0017", "gen_0020", "gen_0025"]
