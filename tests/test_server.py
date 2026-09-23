import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from chaturanga.constants import MANTRI  # noqa: E402
from chaturanga_server.app import app  # noqa: E402

client = TestClient(app)


def test_health_and_meta():
    assert client.get("/api/health").json() == {"status": "ok"}
    meta = client.get("/api/meta").json()
    assert meta["rules"]["setup"] == "crosswise"
    assert {a["id"] for a in meta["agents"]} == {"random", "greedy"}


def test_initial_game():
    g = client.post("/api/game", json={"moves": []}).json()
    assert g["side"] == 1 and g["ply"] == 0
    assert len(g["legal_moves"]) == 16
    assert g["outcome"] is None and g["history"] == []


def test_replayed_moves_and_history():
    g = client.post("/api/game", json={"moves": ["e2e3", "d7d6"]}).json()
    assert g["ply"] == 2 and g["side"] == 1
    assert [h["move"] for h in g["history"]] == ["e2e3", "d7d6"]
    assert g["history"][0]["piece"] == 1 and g["history"][1]["piece"] == -1


def test_illegal_move_is_rejected():
    r = client.post("/api/game", json={"moves": ["e2e4"]})
    assert r.status_code == 400 and "illegal" in r.json()["detail"]


def test_bad_fen_is_rejected():
    assert client.post("/api/game", json={"start_fen": "nonsense"}).status_code == 400


def test_promotion_is_reported_in_history():
    g = client.post("/api/game", json={"start_fen": "7k/2P4p/8/8/8/8/8/K7 w", "moves": ["c7c8"]}).json()
    assert g["history"][0]["promotion"] == MANTRI


def test_checkmate_outcome():
    g = client.post("/api/game", json={"start_fen": "k7/7p/1K6/8/8/8/8/6R1 w", "moves": ["g1g8"]}).json()
    assert g["outcome"] == {"winner": 1, "reason": "checkmate"}
    assert g["legal_moves"] == [] and g["history"][0]["check"] is True


def test_agent_move():
    r = client.post("/api/agent-move", json={"moves": ["e2e3"], "agent": "greedy", "seed": 1}).json()
    assert r["game"]["ply"] == 2
    assert r["game"]["history"][-1]["move"] == r["move"]


def test_agent_move_rejects_unknown_agent_and_finished_games():
    assert client.post("/api/agent-move", json={"agent": "alphazero"}).status_code == 400
    over = {"start_fen": "k7/7p/1K6/8/8/8/8/6R1 w", "moves": ["g1g8"], "agent": "random"}
    assert client.post("/api/agent-move", json=over).status_code == 400
