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
    return {"id": fields.pop("id", "a1b2c3d4e5f6"), "player": "Aditya", "human": "white", "opponent": "zero",
            "level": "hard", "network": NETWORK, "moves": moves, "started": "2026-10-25T10:00:00Z",
            "finished": "2026-10-25T10:20:00Z", **fields}


def test_a_finished_game_is_kept_with_the_engines_result(finished):
    moves, outcome = finished
    r = client.post("/api/human-games", json=game(moves, takebacks=2))
    assert r.status_code == 200, r.text
    body = r.json()
    kept = body["game"]
    assert kept["result"] == ("win" if outcome.winner == 1 else "draw" if outcome.winner is None else "loss")
    assert kept["reason"] == outcome.reason and kept["plies"] == len(moves) and kept["takebacks"] == 2
    assert body["added"] and body["player_games"] == 1 and body["file"].endswith("/aditya.jsonl")
    assert [json.loads(line)["id"] for line in (human_games.games_dir() / "aditya.jsonl").read_text().splitlines()] \
        == ["a1b2c3d4e5f6"]
    # Sent twice (a retry, say), it is kept once.
    again = client.post("/api/human-games", json=game(moves)).json()
    assert not again["added"] and again["player_games"] == 1


def test_resigning_or_leaving_loses_and_the_engine_checks_every_game(finished):
    moves, _ = finished
    for ended in ("resigned", "left"):
        r = client.post("/api/human-games", json=game(moves[:9], id=f"game-{ended}", ended=ended, human="black"))
        assert r.status_code == 200, r.text
        assert r.json()["game"].items() >= {"result": "loss", "reason": ended, "winner": "white"}.items()
    bad = {
        "the game is not over": game(moves[:9]),
        "after the game was over": game(moves, ended="resigned"),
        "before the player made a move": game(moves[:1], ended="resigned", human="black"),
        "is illegal": game(["e2e3", "e2e4"], ended="left"),
        "unknown level": game(moves, level="godlike"),
    }
    for message, body in bad.items():
        r = client.post("/api/human-games", json=body)
        assert r.status_code == 400 and message in r.json()["detail"], (message, r.text)
    assert client.post("/api/human-games", json=game(moves, id="no")).status_code == 422  # ids are 8 or more


def test_results_by_opponent_and_player(finished):
    moves, _ = finished
    plays = [("a", "Aditya", "hard", NETWORK), ("b", "Aditya", "hard", NETWORK), ("c", "Rahul S", "hard", NETWORK),
             ("d", "Rahul S", "easy", NETWORK), ("e", "Rahul S", "hard", {"run": "demo", "generation": 3})]
    for i, (key, player, level, network) in enumerate(plays):
        ended = "resigned" if key == "b" else "over"
        body = game(moves if ended == "over" else moves[:20], id=f"{key}-0000000{i}", player=player, level=level,
                    network=network, ended=ended)
        assert client.post("/api/human-games", json=body).status_code == 200
    client.post("/api/human-games", json=game(moves, id="greedy-001", opponent="greedy", player="Rahul S"))
    s = client.get("/api/human-games").json()
    assert s["total"] == 6 and [g["id"] for g in s["games"]][0] == "greedy-001"
    labels = [(o["label"], o["games"]) for o in s["opponents"]]
    assert labels == [("ChaturangZero Easy, Milestone 4 agent", 1), ("ChaturangZero Hard, Milestone 4 agent", 3),
                      ("ChaturangZero Hard, demo, generation 3", 1), ("Greedy", 1)]
    hard = s["opponents"][1]
    assert (hard["win"], hard["loss"]) == (2, 1) and hard["score"] == pytest.approx(2 / 3)
    assert [(p["player"], p["games"]) for p in hard["players"]] == [("Aditya", 2), ("Rahul S", 1)]
    assert [(p["player"], p["file"], p["games"]) for p in s["players"]] == [("Rahul S", "rahul-s.jsonl", 4),
                                                                            ("Aditya", "aditya.jsonl", 2)]
    assert s["opponents"][3]["level"] is None and s["opponents"][3]["network"] is None


def test_a_teammates_export_imports_once_and_bad_games_are_turned_away(finished, tmp_path, monkeypatch):
    moves, _ = finished
    for i in range(3):
        client.post("/api/human-games", json=game(moves, id=f"teammate-{i:04d}", player="Priya",
                                                   finished=f"2026-10-2{i}T10:00:00Z"))
    r = client.get("/api/human-games/export")
    assert r.status_code == 200 and 'filename="chaturangzero-games-priya-' in r.headers["content-disposition"]
    exported = r.text
    their_file = (human_games.games_dir() / "priya.jsonl").read_text()

    monkeypatch.setenv("CHATURANGA_HUMAN_GAMES", str(tmp_path / "mine"))  # the laptop that collects them
    tampered = json.loads(exported)
    tampered["games"][0]["result"] = "draw" if tampered["games"][0]["result"] != "draw" else "win"
    tampered["games"][1]["moves"] = tampered["games"][1]["moves"][:-1]
    r = client.post("/api/human-games/import", json={"files": [
        {"name": "priya.json", "text": json.dumps(tampered)},
        {"name": "notes.txt", "text": "hello"},
    ]})
    body = r.json()
    assert body["added"] == 1 and body["already"] == 0
    assert [(x["file"], x["game"]) for x in body["rejected"]] == [("priya.json", 1), ("priya.json", 2), ("notes.txt", None)]
    assert "is not what its moves give" in body["rejected"][0]["reason"]

    # The whole export, then the player's own file too: each game once, and the same file as theirs.
    assert client.post("/api/human-games/import", json={"files": [{"name": "p.json", "text": exported}]}).json()["added"] == 2
    again = client.post("/api/human-games/import", json={"files": [{"name": "priya.jsonl", "text": their_file}]}).json()
    assert (again["added"], again["already"], again["total"]) == (0, 3, 3)
    assert (tmp_path / "mine" / "priya.jsonl").read_text() == their_file


def test_player_files_are_named_from_the_name():
    assert human_games.slug("Aditya Bhansali") == "aditya-bhansali"
    assert human_games.slug("  R. K. Sharma!! ") == "r-k-sharma"
    assert human_games.slug("अदिति").startswith("player-")


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
