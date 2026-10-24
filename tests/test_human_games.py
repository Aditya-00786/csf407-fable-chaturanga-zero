"""Recording games people play against the computer, and collecting them from teammates."""

import importlib.util
import json

import pytest

pytest.importorskip("fastapi")
if not (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")):
    pytest.skip("the API tests need httpx2: pip install -e '.[dev]'", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from chaturanga import GameState, move_to_str  # noqa: E402
from chaturanga.agents import MILESTONES, GreedyAgent, RandomAgent  # noqa: E402
from chaturanga.net import NetConfig, new_network, save_checkpoint  # noqa: E402
from chaturanga_server import human_games  # noqa: E402
from chaturanga_server.app import app  # noqa: E402

client = TestClient(app)
NETWORK = {"run": "main-v4", "generation": 100, "fingerprint": next(iter(MILESTONES))}


@pytest.fixture(scope="module")
def finished():
    """The moves of a finished game: Greedy (White, the person here) against Random."""
    state, moves, players = GameState.initial(), [], {1: GreedyAgent(seed=3), -1: RandomAgent(seed=3)}
    while not state.is_terminal():
        move = players[state.side].select_move(state)
        moves.append(move_to_str(move))
        state = state.apply(move)
    return moves, state.outcome()


def game(moves, **fields):
    return {"id": fields.pop("id", "a1b2c3d4e5f6"), "human": "white", "opponent": "zero", "level": "hard",
            "network": NETWORK, "moves": moves, **fields}


def test_a_finished_game_is_kept_with_the_engines_result_and_nothing_about_the_person(finished):
    moves, outcome = finished
    r = client.post("/api/human-games", json=game(moves, takebacks=2, player="Aditya", started="2026-10-25T10:00:00Z"))
    assert r.status_code == 200, r.text
    body = r.json()
    kept = body["game"]
    assert kept["result"] == ("win" if outcome.winner == 1 else "draw" if outcome.winner is None else "loss")
    assert kept["reason"] == outcome.reason and kept["moves"] == moves and kept["takebacks"] == 2
    # Only what the results need: no name, no times, nothing worked out from the rest.
    assert set(kept) == {"id", "human", "opponent", "level", "network", "result", "reason", "takebacks", "moves"}
    assert body["added"] and body["total"] == 1
    assert json.loads((human_games.games_dir() / "a1b2c3d4e5f6.json").read_text()) == kept
    # Sent twice (a retry, say), it is kept once.
    again = client.post("/api/human-games", json=game(moves)).json()
    assert not again["added"] and again["total"] == 1


def test_resigning_or_leaving_loses_and_the_engine_checks_every_game(finished):
    moves, _ = finished
    for ended in ("resigned", "left"):
        r = client.post("/api/human-games", json=game(moves[:9], id=f"game-{ended}", ended=ended, human="black"))
        assert r.status_code == 200, r.text
        assert r.json()["game"].items() >= {"result": "loss", "reason": ended}.items()
    bad = {
        "the game is not over": game(moves[:9]),
        "after the game was over": game(moves, ended="resigned"),
        "before the person made a move": game(moves[:1], ended="resigned", human="black"),
        "is illegal": game(["e2e3", "e2e4"], ended="left"),
        "unknown level": game(moves, level="godlike"),
    }
    for message, body in bad.items():
        r = client.post("/api/human-games", json=body)
        assert r.status_code == 400 and message in r.json()["detail"], (message, r.text)
    assert client.post("/api/human-games", json=game(moves, id="no")).status_code == 422  # ids are 8 or more
    assert client.post("/api/human-games", json=game(moves, id="../../etc")).status_code == 422  # and name a file


def test_results_by_opponent(finished):
    moves, _ = finished
    plays = [("a", "hard", NETWORK), ("b", "hard", NETWORK), ("c", "hard", NETWORK), ("d", "easy", NETWORK),
             ("e", "hard", {"run": "demo", "generation": 3})]
    for i, (key, level, network) in enumerate(plays):
        ended = "resigned" if key == "b" else "over"
        body = game(moves if ended == "over" else moves[:20], id=f"{key}-0000000{i}", level=level, network=network,
                    ended=ended)
        assert client.post("/api/human-games", json=body).status_code == 200
    client.post("/api/human-games", json=game(moves, id="greedy-001", opponent="greedy"))
    s = client.get("/api/human-games").json()
    assert s["total"] == 6 and s["games"][0]["id"] == "greedy-001"  # the most recently added first
    labels = [(o["label"], o["games"]) for o in s["opponents"]]
    assert labels == [("ChaturangZero Easy, Milestone 4 agent", 1), ("ChaturangZero Hard, Milestone 4 agent", 3),
                      ("ChaturangZero Hard, demo, generation 3", 1), ("Greedy", 1)]
    hard = s["opponents"][1]
    assert (hard["win"], hard["loss"]) == (2, 1) and hard["score"] == pytest.approx(2 / 3)
    assert s["opponents"][3]["level"] is None and s["opponents"][3]["network"] is None


def test_a_teammates_export_imports_once_and_bad_games_are_turned_away(finished, tmp_path, monkeypatch):
    moves, _ = finished
    for i in range(3):
        client.post("/api/human-games", json=game(moves, id=f"teammate-{i:04d}"))
    r = client.get("/api/human-games/export")
    assert r.status_code == 200 and 'filename="chaturangzero-games.json"' in r.headers["content-disposition"]
    exported = r.text
    their_files = {p.name: p.read_text() for p in human_games.games_dir().glob("*.json")}

    monkeypatch.setenv("CHATURANGA_HUMAN_GAMES", str(tmp_path / "mine"))  # the laptop that collects them
    tampered = json.loads(exported)
    tampered["games"][0]["result"] = "draw" if tampered["games"][0]["result"] != "draw" else "win"
    tampered["games"][1]["moves"] = tampered["games"][1]["moves"][:-1]
    r = client.post("/api/human-games/import", json={"files": [
        {"name": "theirs.json", "text": json.dumps(tampered)},
        {"name": "notes.txt", "text": "hello"},
    ]})
    body = r.json()
    assert body["added"] == 1 and body["already"] == 0
    assert [(x["file"], x["game"]) for x in body["rejected"]] == [("theirs.json", 1), ("theirs.json", 2), ("notes.txt", None)]
    assert "is not what its moves give" in body["rejected"][0]["reason"]

    # The whole export, then a game's own file: each game once, and the same files as theirs.
    assert client.post("/api/human-games/import", json={"files": [{"name": "t.json", "text": exported}]}).json()["added"] == 2
    name, text = next(iter(their_files.items()))
    again = client.post("/api/human-games/import", json={"files": [{"name": name, "text": text}]}).json()
    assert (again["added"], again["already"], again["total"]) == (0, 1, 3)
    assert {p.name: p.read_text() for p in (tmp_path / "mine").glob("*.json")} == their_files


def test_a_name_or_times_in_a_file_from_elsewhere_are_left_out(finished):
    moves, outcome = finished
    old = {**game(moves, id="older-format-1"), "player": "Rahul", "finished": "2026-10-25T10:20:00Z",
           "result": "win" if outcome.winner == 1 else "loss", "reason": outcome.reason}
    r = client.post("/api/human-games/import", json={"files": [{"name": "old.jsonl", "text": json.dumps(old)}]})
    assert r.json()["added"] == 1
    kept = json.loads((human_games.games_dir() / "older-format-1.json").read_text())
    assert "player" not in kept and "finished" not in kept


def test_a_network_file_chosen_in_the_app_is_played_with(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "old" / "checkpoints").mkdir(parents=True)
    (runs / "old" / "config.json").write_text("{}")
    (runs / "old" / "launch.json").write_text(json.dumps({"play": "best", "where": "mac"}))
    monkeypatch.setenv("CHATURANGA_RUNS", str(runs))
    target = tmp_path / "checkpoints" / "best.pt"
    monkeypatch.setenv("CHATURANGA_CHECKPOINT", str(target))
    file = tmp_path / "net.pt"
    save_checkpoint(str(file), new_network(NetConfig(blocks=1, channels=16), seed=5), generation=12, run="shared")

    r = client.post("/api/training/play/file?name=junk.pt", content=b"not a network")
    assert r.status_code == 400 and "junk.pt" in r.json()["detail"] and not target.exists()

    r = client.post("/api/training/play/file?name=net.pt", content=file.read_bytes())
    assert r.status_code == 200, r.text
    info = r.json()
    assert (info["run"], info["generation"], info["milestone"]) == ("shared", 12, None)
    assert target.read_bytes() == file.read_bytes()
    assert client.get("/api/meta").json()["trained"]["fingerprint"] == info["fingerprint"]
    # The run that gave Play its network no longer does, so the Training page does not put it back.
    assert json.loads((runs / "old" / "launch.json").read_text())["play"] is None
