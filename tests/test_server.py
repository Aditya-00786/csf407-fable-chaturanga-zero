import importlib.util

import pytest

pytest.importorskip("fastapi")
# Starlette's test client uses httpx2 (the successor to httpx), or httpx on older versions.
if not (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")):
    pytest.skip("the API tests need httpx2: pip install -e '.[dev]'", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from chaturanga.constants import MANTRI  # noqa: E402
from chaturanga_server.app import app  # noqa: E402

client = TestClient(app)


def test_health_and_meta():
    assert client.get("/api/health").json() == {"status": "ok"}
    meta = client.get("/api/meta").json()
    assert meta["rules"]["setup"] == "crosswise"
    assert {a["id"] for a in meta["agents"]} == {"random", "greedy", "mcts"}


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


def test_mcts_agent_move_is_legal_and_takes_a_free_win():
    r = client.post("/api/agent-move", json={"moves": [], "agent": "mcts", "seed": 0}).json()
    assert r["game"]["ply"] == 1 and r["game"]["history"][0]["move"] == r["move"]
    trap = {"start_fen": "7k/7p/4K2P/8/8/8/8/R7 w", "moves": [], "agent": "mcts"}
    assert client.post("/api/agent-move", json=trap).json()["move"] == "a1a8"


def test_agent_move_rejects_unknown_agent_and_finished_games():
    assert client.post("/api/agent-move", json={"agent": "alphazero"}).status_code == 400
    over = {"start_fen": "k7/7p/1K6/8/8/8/8/6R1 w", "moves": ["g1g8"], "agent": "random"}
    assert client.post("/api/agent-move", json=over).status_code == 400


def test_startup_loads_the_mcts_network_in_the_background():
    import threading
    import time

    from chaturanga import agents

    with TestClient(app) as started:  # runs the app's start-up hook
        assert started.get("/api/health").json() == {"status": "ok"}  # answers while loading
        for thread in threading.enumerate():
            if thread.name == "mcts-warm-up":
                thread.join(timeout=60)
        deadline = time.time() + 60
        while agents._evaluator is None and time.time() < deadline:
            time.sleep(0.05)
        assert agents._evaluator is not None


def test_mcts_reports_a_failed_network_load_clearly(monkeypatch):
    from chaturanga import agents

    monkeypatch.setattr(agents, "_evaluator", None)
    monkeypatch.setattr(agents, "_load_error", "reading PyTorch's files from disk timed out")
    r = client.post("/api/agent-move", json={"moves": [], "agent": "mcts"})
    assert r.status_code == 503
    assert "could not be loaded" in r.json()["detail"] and "Restart the API server" in r.json()["detail"]
    # The other computer players are unaffected.
    assert client.post("/api/agent-move", json={"moves": [], "agent": "greedy"}).status_code == 200


def test_mcts_does_not_wait_forever_for_a_slow_load(monkeypatch):
    import threading

    import pytest

    from chaturanga import agents

    monkeypatch.setattr(agents, "_evaluator", None)
    monkeypatch.setattr(agents, "_evaluator_lock", threading.Lock())
    monkeypatch.setattr(agents, "NETWORK_WAIT_SECONDS", 0.2)
    agents._evaluator_lock.acquire()  # another thread is "still loading"
    try:
        with pytest.raises(agents.NetworkUnavailable, match="still loading"):
            agents.make_agent("mcts")
        r = client.post("/api/agent-move", json={"moves": [], "agent": "mcts"})
        assert r.status_code == 503 and "still loading" in r.json()["detail"]
    finally:
        agents._evaluator_lock.release()
