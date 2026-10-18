"""Milestone 3 in the app: the trained agent with its levels, and the Watch training API."""

import importlib.util
import os
import pytest

pytest.importorskip("fastapi")
if not (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")):
    pytest.skip("the API tests need httpx2: pip install -e '.[dev]'", allow_module_level=True)

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from chaturanga import agents  # noqa: E402
from chaturanga.net import NetConfig, new_network, save_checkpoint  # noqa: E402
from chaturanga.training import TrainConfig, TrainingRun  # noqa: E402
from chaturanga_server.app import app  # noqa: E402

client = TestClient(app)
TINY = NetConfig(blocks=1, channels=16)


@pytest.fixture
def no_trained_network(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_CHECKPOINT", str(tmp_path / "missing.pt"))


@pytest.fixture
def trained_network(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    save_checkpoint(str(path), new_network(TINY, seed=1), generation=7, run="demo")
    monkeypatch.setenv("CHATURANGA_CHECKPOINT", str(path))
    return path


def test_meta_lists_the_trained_agent_and_its_levels(trained_network):
    meta = client.get("/api/meta").json()
    assert {a["id"] for a in meta["agents"]} == {"random", "greedy", "mcts", "zero"}
    assert [lv["id"] for lv in meta["levels"]] == ["easy", "medium", "hard", "impossible"]
    assert meta["trained"] == {"run": "demo", "generation": 7}


def test_without_a_trained_network_the_agent_explains(no_trained_network):
    assert client.get("/api/meta").json()["trained"] is None
    r = client.post("/api/agent-move", json={"moves": [], "agent": "zero", "level": "easy"})
    assert r.status_code == 503 and "no trained network" in r.json()["detail"]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_trained_agent_plays_at_each_level(trained_network, level):
    r = client.post("/api/agent-move", json={"moves": ["e2e3"], "agent": "zero", "level": level, "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["game"]["ply"] == 2 and body["network"] == {"run": "demo", "generation": 7}


def test_unknown_level_is_a_bad_request(trained_network):
    r = client.post("/api/agent-move", json={"moves": [], "agent": "zero", "level": "godlike"})
    assert r.status_code == 400


def test_levels_search_more_as_they_get_harder():
    sims = [lv.simulations for lv in agents.LEVELS.values()]
    temps = [lv.temperature for lv in agents.LEVELS.values()]
    assert sims == sorted(sims) and len(set(sims)) == 4 and temps == sorted(temps, reverse=True)


def test_a_newly_published_network_is_picked_up(trained_network):
    assert agents.trained_evaluator()[1]["generation"] == 7
    save_checkpoint(str(trained_network), new_network(TINY, seed=2), generation=8, run="demo")
    os.utime(trained_network, ns=(1, 2_000_000_000_000_000_000))  # a new modification time, however fast the test
    assert agents.trained_evaluator()[1]["generation"] == 8


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    cfg = TrainConfig(generations=1, games_per_generation=2, simulations=4, concurrent=2, steps_per_generation=12,
                      batch_size=16, eval_games=2, eval_opponents=("previous",), blocks=1, channels=16)
    TrainingRun(str(root / "demo"), cfg, workers=0, device=torch.device("cpu"), log=lambda *_: None).run()
    return root


def test_training_runs_are_listed(run_root, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(run_root))
    runs = client.get("/api/training/runs").json()["runs"]
    assert [r["name"] for r in runs] == ["demo"]
    assert runs[0]["state"] == "finished" and runs[0]["live"] is False


def test_run_detail_has_losses_and_generations(run_root, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(run_root))
    d = client.get("/api/training/runs/demo").json()
    assert d["config"]["games_per_generation"] == 2
    assert [p["step"] for p in d["train"]] == [10, 12]
    (gen,) = d["generations"]
    assert gen["generation"] == 1 and gen["games"] == 2 and "previous" in gen["eval"]
    assert set(gen["held_out"]) == {"loss", "value_loss", "policy_loss", "value_baseline", "step"}
    assert gen["best"] in (0, 1) and gen["steps"] == 12


def test_self_play_games_can_be_fetched(run_root, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(run_root))
    games = client.get("/api/training/runs/demo/games/1").json()["games"]
    assert len(games) == 2
    g = client.get("/api/training/runs/demo/games/1/1").json()
    assert len(g["moves"]) == games[1]["plies"] and g["network"] == 0
    # The moves replay as a legal game through the ordinary game endpoint.
    view = client.post("/api/game", json={"moves": g["moves"]}).json()
    assert view["outcome"]["reason"] == g["reason"]


@pytest.mark.parametrize("url", ["/api/training/runs/nope", "/api/training/runs/..", "/api/training/runs/demo/games/9",
                                 "/api/training/runs/demo/games/1/99"])
def test_missing_runs_and_games_are_404(run_root, monkeypatch, url):
    monkeypatch.setenv("CHATURANGA_RUNS", str(run_root))
    assert client.get(url).status_code == 404


def test_loss_curve_is_downsampled():
    from chaturanga_server.training_api import _downsample

    points = [{"step": i, "generation": 1, "loss": float(i)} for i in range(1, 1001)]
    out = _downsample(points, ("loss",), limit=100)
    assert len(out) == 100 and out[-1]["step"] == 1000 and out[0]["loss"] == pytest.approx(5.5)


def test_runs_are_unaffected_by_unrelated_folders(tmp_path, monkeypatch):
    (tmp_path / "notes").mkdir()
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    assert client.get("/api/training/runs").json() == {"runs": []}


# ---------------------------------------------------------------------- pausing and resuming

def _fake_run(root, name="r", **status):
    """A run directory with just a config and a status, as the API sees it."""
    import json
    import socket

    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"generations": 3}))
    base = {"run": name, "generation": 1, "generations": 3, "host": socket.gethostname(),
            "updated": "2026-01-01T00:00:00+00:00"}
    (d / "status.json").write_text(json.dumps({**base, **status}))
    return d


def test_a_run_whose_process_is_alive_is_live_however_quiet(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _fake_run(tmp_path, state="selfplay", pid=os.getpid())
    (run,) = client.get("/api/training/runs").json()["runs"]
    assert run["live"] is True


def test_a_run_whose_process_died_is_not_live(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _fake_run(tmp_path, state="selfplay", pid=2**22 + 12345)  # no such process
    (run,) = client.get("/api/training/runs").json()["runs"]
    assert run["live"] is False


def test_pause_writes_a_request_the_run_will_read(tmp_path, monkeypatch):
    from chaturanga.training import pending_pause

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    d = _fake_run(tmp_path, state="training", pid=os.getpid())
    r = client.post("/api/training/runs/r/pause", json={"when": "after_phase"})
    assert r.status_code == 200 and r.json()["pause_requested"] == "after_phase"
    assert pending_pause(d) == "after_phase"
    assert client.post("/api/training/runs/r/pause", json={"when": None}).json()["pause_requested"] is None
    assert client.post("/api/training/runs/r/pause", json={"when": "soon"}).status_code == 422


def test_cannot_pause_a_run_that_is_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _fake_run(tmp_path, state="paused", pid=os.getpid())
    assert client.post("/api/training/runs/r/pause", json={"when": "now"}).status_code == 409


def test_resume_starts_the_run_detached(tmp_path, monkeypatch):
    from chaturanga_server import training_api

    started = {}

    class FakeProcess:
        pid = os.getpid()  # "alive", so the run now reads as starting and live

        def wait(self):
            started["waited"] = True

    def fake_popen(cmd, **kwargs):
        started.update(cmd=cmd, **kwargs)
        return FakeProcess()

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    monkeypatch.setattr(training_api.subprocess, "Popen", fake_popen)
    _fake_run(tmp_path, state="paused", pid=2**22 + 12345)
    r = client.post("/api/training/runs/r/resume", json={})
    assert r.status_code == 202 and r.json()["state"] == "starting" and r.json()["live"] is True
    cmd = started["cmd"]
    assert cmd[cmd.index("--run") + 1] == "r" and "--generations" not in cmd and started["start_new_session"]
    import time

    for _ in range(100):  # the server collects the process when it ends (no zombie), from a thread
        if started.get("waited"):
            break
        time.sleep(0.01)
    assert started.get("waited")
    # Now it is running, so a second resume is refused.
    assert client.post("/api/training/runs/r/resume", json={}).status_code == 409


def test_a_finished_run_resumes_only_to_train_further(tmp_path, monkeypatch):
    from chaturanga_server import training_api

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    monkeypatch.setattr(training_api.subprocess, "Popen",
                        lambda cmd, **kw: type("P", (), {"pid": os.getpid(), "wait": lambda self: None})())
    d = _fake_run(tmp_path, state="finished", pid=2**22 + 12345)
    for g in range(4):
        (d / "checkpoints" / f"gen_{g:04d}.pt").write_bytes(b"")
    assert client.post("/api/training/runs/r/resume", json={}).status_code == 400
    r = client.post("/api/training/runs/r/resume", json={"generations": 5})
    assert r.status_code == 202 and r.json()["generations"] == 5
