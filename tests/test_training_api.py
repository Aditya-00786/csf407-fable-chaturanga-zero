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
    cfg = TrainConfig(generations=1, games_per_generation=2, simulations=4, concurrent=2, steps_per_generation=12, endgame_fraction=0,
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


def _capture_popen(monkeypatch):
    from chaturanga_server import training_api

    started = {}

    def fake_popen(cmd, **kwargs):
        started["cmd"] = cmd
        return type("P", (), {"pid": os.getpid(), "wait": lambda self: None})()

    monkeypatch.setattr(training_api.subprocess, "Popen", fake_popen)
    return started


def test_start_a_new_run_from_the_app(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    started = _capture_popen(monkeypatch)
    r = client.post("/api/training/runs", json={"name": "quick", "generations": 3, "games_per_generation": 8,
                                                 "simulations": 20, "eval_games": 4, "publish": False,
                                                 "auto_report": True})
    assert r.status_code == 202 and r.json()["state"] == "starting" and r.json()["live"] is True
    config = json.loads((tmp_path / "quick" / "config.json").read_text())
    assert (config["generations"], config["games_per_generation"], config["simulations"], config["eval_games"]) == (3, 8, 20, 4)
    assert config["auto_report"] is True
    assert config["buffer_positions"] == 60_000 and config["value_mix"] == 0.5
    cmd = started["cmd"]
    assert cmd[cmd.index("--run") + 1] == "quick" and cmd[cmd.index("--publish") + 1] == ""  # not published
    assert [r["name"] for r in client.get("/api/training/runs").json()["runs"]] == ["quick"]


def test_a_run_with_many_games_gets_a_buffer_to_match(tmp_path, monkeypatch):
    """About the last three generations: more games than the value head can learn by heart."""
    import json

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _capture_popen(monkeypatch)
    assert client.post("/api/training/runs", json={"name": "big", "games_per_generation": 500}).status_code == 202
    config = json.loads((tmp_path / "big" / "config.json").read_text())
    assert config["buffer_positions"] == 300_000 and config["steps_per_generation"] >= 4 * 500 * 200 // 256


def test_a_new_run_can_have_a_larger_network(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _capture_popen(monkeypatch)
    assert client.post("/api/training/runs", json={"name": "small"}).status_code == 202
    (tmp_path / "small" / "status.json").write_text(json.dumps({"state": "finished"}))  # one run at a time here
    assert client.post("/api/training/runs", json={"name": "large", "blocks": 10, "channels": 128}).status_code == 202
    sizes = {n: json.loads((tmp_path / n / "config.json").read_text()) for n in ("small", "large")}
    assert (sizes["small"]["blocks"], sizes["small"]["channels"]) == (6, 64)
    assert (sizes["large"]["blocks"], sizes["large"]["channels"]) == (10, 128)
    assert client.post("/api/training/runs", json={"name": "huge", "channels": 4096}).status_code == 422


def test_a_new_run_needs_a_fresh_valid_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _capture_popen(monkeypatch)
    _fake_run(tmp_path, "main", state="paused", pid=2**22 + 12345)
    assert client.post("/api/training/runs", json={"name": "main"}).status_code == 409
    assert client.post("/api/training/runs", json={"name": "../evil"}).status_code == 400
    assert client.post("/api/training/runs", json={"name": "-x"}).status_code == 400
    assert client.post("/api/training/runs", json={"name": "ok", "games_per_generation": 1}).status_code == 422


def test_one_run_trains_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _capture_popen(monkeypatch)
    _fake_run(tmp_path, "main", state="selfplay", pid=os.getpid())  # training now
    _fake_run(tmp_path, "old", state="paused", pid=2**22 + 12345)
    r = client.post("/api/training/runs", json={"name": "second"})
    assert r.status_code == 409 and "'main' is training" in r.json()["detail"]
    assert not (tmp_path / "second").exists()
    assert client.post("/api/training/runs/old/resume", json={}).status_code == 409


def test_resume_keeps_a_run_unpublished_if_it_started_that_way(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    started = _capture_popen(monkeypatch)
    d = _fake_run(tmp_path, "test", state="paused", pid=2**22 + 12345)
    (d / "launch.json").write_text('{"publish": false}')
    assert client.post("/api/training/runs/test/resume", json={}).status_code == 202
    assert started["cmd"][started["cmd"].index("--publish") + 1] == ""


def test_delete_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    d = _fake_run(tmp_path, "old", state="paused", pid=2**22 + 12345)
    (d / "games").mkdir()
    (d / "games" / "gen_0001.jsonl.gz").write_bytes(b"x")
    keep = _fake_run(tmp_path, "keep", state="finished")
    assert client.delete("/api/training/runs/old").json() == {"deleted": "old"}
    assert not d.exists() and keep.exists()
    assert client.delete("/api/training/runs/old").status_code == 404


def test_a_training_run_cannot_be_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    d = _fake_run(tmp_path, "main", state="selfplay", pid=os.getpid())
    r = client.delete("/api/training/runs/main")
    assert r.status_code == 409 and "Pause it" in r.json()["detail"] and d.exists()


def test_delete_stays_inside_the_runs_folder(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = _fake_run(tmp_path, "precious", state="finished")
    (runs / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("CHATURANGA_RUNS", str(runs))
    assert client.delete("/api/training/runs/link").status_code == 400
    assert client.delete("/api/training/runs/..").status_code in (404, 405)
    assert outside.exists() and (outside / "config.json").exists()


# ---------------------------------------------------------------------- the run report

def test_a_report_is_made_only_when_asked(run_root, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(run_root))
    assert client.get("/api/training/runs/demo/report").json() == {"report": None, "strength": None}
    assert not (run_root / "demo" / "report" / "report.json").exists()
    r = client.post("/api/training/runs/demo/report").json()["report"]
    s = r["summary"]
    assert r["generations"] == 1 and s["policy"]["verdict"] in ("learned", "not_learned")
    assert s["value"]["generations"] == 1 and s["strength"]["previous"]["last_generation"] == 1
    probe = r["probe"]
    assert len(probe["positions"]) == 6 and [n["label"] for n in probe["networks"]] == ["untrained", "final"]
    assert all(len(n["values"]) == 6 and -1 <= min(n["values"]) <= max(n["values"]) <= 1 for n in probe["networks"])
    # Saved with the run, and read back as it was.
    assert client.get("/api/training/runs/demo/report").json()["report"] == r


def test_no_report_while_the_run_trains(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    _fake_run(tmp_path, "main", state="training", pid=os.getpid())
    assert client.post("/api/training/runs/main/report").status_code == 409


def test_strength_check_plays_every_network_against_both_baselines(run_root):
    from chaturanga.analysis import measure_strength, read_strength

    data = measure_strength(run_root / "demo", games=2, workers=1, simulations=4, log=lambda *_: None)
    assert data["state"] == "done" and data["done"] == data["total"] == 2 * 2 * 2  # 2 networks x 2 opponents x 2
    for label in ("untrained", "final"):
        for opponent in ("random", "greedy"):
            r = data["results"][label][opponent]
            assert r["wins"] + r["draws"] + r["losses"] == 2 and 0 <= r["score"] <= 1
    assert read_strength(run_root / "demo")["state"] == "done"


def test_strength_check_starts_in_the_background_and_waits_for_training(tmp_path, monkeypatch, run_root):
    import shutil

    monkeypatch.setenv("CHATURANGA_RUNS", str(tmp_path))
    shutil.copytree(run_root / "demo", tmp_path / "demo")
    started = _capture_popen(monkeypatch)
    _fake_run(tmp_path, "main", state="training", pid=os.getpid())
    assert client.post("/api/training/runs/demo/report/strength", json={"games": 10}).status_code == 409
    (tmp_path / "main" / "status.json").write_text('{"state": "paused"}')
    r = client.post("/api/training/runs/demo/report/strength", json={"games": 10})
    assert r.status_code == 202 and r.json()["state"] == "running"
    cmd = started["cmd"]
    assert "--strength-only" in cmd and cmd[cmd.index("--games") + 1] == "10"
    assert client.post("/api/training/runs/demo/report/strength", json={}).status_code == 409  # already running


def test_run_charts_are_drawn_as_an_image(run_root, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("CHATURANGA_RUNS", str(run_root))
    r = client.get("/api/training/runs/demo/charts.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
