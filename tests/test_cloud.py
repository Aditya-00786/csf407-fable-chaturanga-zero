"""Training in the cloud: Colab through Google Drive, and Kaggle through its command-line tool.

No test touches a real Drive or Kaggle account: Drive is a folder here, and Kaggle a small fake
of its tool that keeps its state in files (see conftest.py's no_real_cloud).
"""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
if not (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")):
    pytest.skip("the API tests need httpx2: pip install -e '.[dev]'", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from chaturanga import cloud  # noqa: E402
from chaturanga.training import TrainConfig, TrainingRun  # noqa: E402
from chaturanga_server.app import app  # noqa: E402

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
TINY = {"generations": 1, "games_per_generation": 2, "simulations": 4, "eval_games": 2}


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Local runs, a Drive folder, and where the published network goes."""
    runs, drive = tmp_path / "runs", tmp_path / "drive" / "ChaturangZero"
    runs.mkdir()
    monkeypatch.setenv("CHATURANGA_RUNS", str(runs))
    monkeypatch.setenv("CHATURANGA_CHECKPOINT", str(tmp_path / "published" / "best.pt"))
    return {"runs": runs, "drive": drive, "published": tmp_path / "published" / "best.pt", "tmp": tmp_path}


def _no_spawn(monkeypatch):
    from chaturanga_server import training_api

    started = []
    real_popen = training_api.subprocess.Popen

    def fake_popen(cmd, *args, **kwargs):
        if not any(str(c).endswith(("train.py", "kaggle_run.py", "report.py")) for c in cmd):
            return real_popen(cmd, *args, **kwargs)  # git and the like run for real
        started.append(cmd)
        return type("P", (), {"pid": os.getpid(), "wait": lambda self: None})()

    monkeypatch.setattr(training_api.subprocess, "Popen", fake_popen)
    return started


# ---------------------------------------------------------------------- Colab

def test_colab_needs_google_drive_here(places, monkeypatch):
    _no_spawn(monkeypatch)
    r = client.post("/api/training/runs", json={"name": "c1", "where": "colab", **TINY})
    assert r.status_code == 409 and "Google Drive for desktop" in r.json()["detail"]
    assert client.get("/api/training/cloud").json()["colab"]["drive"] is None


def test_a_colab_run_waits_in_drive_for_the_notebook(places, monkeypatch):
    monkeypatch.setenv("CHATURANGA_DRIVE", str(places["drive"]))
    started = _no_spawn(monkeypatch)
    r = client.post("/api/training/runs", json={"name": "c1", "where": "colab", "publish": True, **TINY})
    assert r.status_code == 202 and not started  # nothing runs on this Mac
    body = r.json()
    assert body["state"] == "waiting" and body["where"] == "colab" and body["live"] is False
    assert body["colab_url"].startswith("https://colab.research.google.com/github/") and "train_colab.ipynb" in body["colab_url"]
    run = places["drive"] / "runs" / "c1"
    assert json.loads((run / "config.json").read_text())["generations"] == 1
    # The run is listed with the local ones, marked as a Colab run.
    runs = client.get("/api/training/runs").json()["runs"]
    assert [(x["name"], x["where"]) for x in runs] == [("c1", "colab")]
    # The notebook takes the request once.
    taken = cloud.take_request(places["drive"])
    assert taken["run"] == "c1" and (places["drive"] / "requests" / "c1.taken").exists()
    assert cloud.take_request(places["drive"]) is None
    # A Mac run with the same name is refused: names are shared across places.
    assert client.post("/api/training/runs", json={"name": "c1", **TINY}).status_code == 409


def test_a_colab_run_in_drive_trains_and_reports_like_a_local_one(places, monkeypatch):
    monkeypatch.setenv("CHATURANGA_DRIVE", str(places["drive"]))
    _no_spawn(monkeypatch)
    client.post("/api/training/runs", json={"name": "c2", "where": "colab", "publish": True, **TINY})
    run = places["drive"] / "runs" / "c2"
    assert cloud.take_request(places["drive"])["run"] == "c2"
    # What the notebook then does, in Drive: train the run from its folder.
    TrainingRun(str(run), TrainConfig(**json.loads((run / "config.json").read_text())), workers=0,
                log=lambda *_: None).run()
    detail = client.get("/api/training/runs/c2").json()
    assert detail["status"]["state"] == "finished" and detail["status"]["where"] == "colab"
    # Its best network reaches the Trained opponent through Drive.
    client.get("/api/training/runs")
    assert places["published"].exists()
    assert client.post("/api/training/runs/c2/report").status_code == 200


def test_pausing_a_colab_run_writes_to_drive_and_resume_asks_colab_again(places, monkeypatch):
    monkeypatch.setenv("CHATURANGA_DRIVE", str(places["drive"]))
    _no_spawn(monkeypatch)
    client.post("/api/training/runs", json={"name": "c3", "where": "colab", **TINY})
    run = places["drive"] / "runs" / "c3"
    cloud.take_request(places["drive"])
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run / "status.json").write_text(json.dumps({"state": "selfplay", "host": "colab-vm", "pid": 1, "updated": now}))
    assert client.post("/api/training/runs/c3/pause", json={"when": "after_phase"}).status_code == 200
    assert json.loads((run / "control.json").read_text())["pause"] == "after_phase"
    (run / "status.json").write_text(json.dumps({"state": "paused", "updated": now}))
    r = client.post("/api/training/runs/c3/resume", json={})
    assert r.status_code == 202 and r.json()["state"] == "waiting" and "colab_url" in r.json()
    assert (places["drive"] / "requests" / "c3.json").exists()


# ---------------------------------------------------------------------- Kaggle

FAKE_KAGGLE = r'''#!{python}
"""A stand-in for the kaggle tool: state in files under FAKE_KAGGLE_STATE."""
import json, os, shutil, sys
from pathlib import Path
state = Path(os.environ["FAKE_KAGGLE_STATE"])
args = sys.argv[1:]
token = os.environ.get("KAGGLE_API_TOKEN", "")
if token and Path(token).exists():
    token = Path(token).read_text().strip()
if token != "good-token":
    print("Authentication required to call the Kaggle API."); sys.exit(1)
cmd = " ".join(args[:2])
if cmd == "kernels list":
    print("ref title"); sys.exit(0)
if cmd == "datasets status":
    if (state / "dataset").exists():
        print("ready"); sys.exit(0)
    print("404 Client Error: Not Found"); sys.exit(1)
if cmd in ("datasets create", "datasets version"):
    folder = Path(args[args.index("-p") + 1])
    shutil.copy(folder / "bundle.tar.gz", state / "bundle.tar.gz")
    (state / "dataset").write_text(cmd); print("ok"); sys.exit(0)
if cmd == "kernels push":
    folder = Path(args[args.index("-p") + 1])
    shutil.copy(folder / "run.py", state / "run.py")
    shutil.copy(folder / "kernel-metadata.json", state / "kernel-metadata.json")
    (state / "checks").write_text("0"); print("pushed"); sys.exit(0)
if cmd == "kernels status":
    n = int((state / "checks").read_text()); (state / "checks").write_text(str(n + 1))
    word = ["QUEUED", "RUNNING", "RUNNING", os.environ.get("FAKE_KAGGLE_END", "COMPLETE")][min(n, 3)]
    print(f'{{args[2]}} has status "KernelWorkerStatus.{{word}}"'); sys.exit(0)
if cmd == "kernels output":
    out = Path(args[args.index("-p") + 1])
    if (state / "working").exists():  # (a session stopped on Kaggle leaves nothing)
        shutil.copytree(state / "working" / "runs", out / "runs", dirs_exist_ok=True)
    (out / "session.log").write_text("all good"); sys.exit(0)
print("unknown", args); sys.exit(2)
'''


@pytest.fixture
def fake_kaggle(places, monkeypatch):
    state = places["tmp"] / "kaggle-state"
    state.mkdir()
    tool = places["tmp"] / "kaggle"
    tool.write_text(FAKE_KAGGLE.format(python=sys.executable))
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CHATURANGA_KAGGLE_TOOL", str(tool))
    monkeypatch.setenv("FAKE_KAGGLE_STATE", str(state))
    return state


def test_connecting_kaggle_checks_the_token_and_keeps_it_private(places, fake_kaggle):
    assert client.get("/api/training/cloud").json()["kaggle"]["connected"] is False
    bad = client.post("/api/training/cloud/kaggle", json={"username": "me", "token": "wrong-token"})
    assert bad.status_code == 400 and "did not accept" in bad.json()["detail"]
    ok = client.post("/api/training/cloud/kaggle", json={"username": "me", "token": "good-token"})
    assert ok.status_code == 200 and ok.json()["kaggle"] == {"connected": True, "username": "me", "tool": True}
    token_file = cloud.kaggle_dir() / "access_token"
    assert token_file.read_text() == "good-token" and stat.S_IMODE(token_file.stat().st_mode) == 0o600


def test_a_kaggle_run_needs_a_connection_and_starts_its_follower(places, fake_kaggle, monkeypatch):
    started = _no_spawn(monkeypatch)
    r = client.post("/api/training/runs", json={"name": "k1", "where": "kaggle", **TINY})
    assert r.status_code == 409 and "Connect Kaggle" in r.json()["detail"]
    cloud.save_kaggle_credentials("me", "good-token")
    r = client.post("/api/training/runs", json={"name": "k1", "where": "kaggle", "publish": True, **TINY})
    assert r.status_code == 202 and r.json()["state"] == "uploading" and r.json()["where"] == "kaggle"
    (cmd,) = started
    assert cmd[1:3] == ["-u", str(ROOT / "scripts" / "kaggle_run.py")] and cmd[cmd.index("--run") + 1] == "k1"
    assert cmd[cmd.index("--publish") + 1] == str(places["published"])
    # Kaggle runs are paused on Kaggle, not from here; they do not block a run on this Mac.
    assert client.post("/api/training/runs/k1/pause", json={"when": "now"}).status_code == 409
    assert client.post("/api/training/runs", json={"name": "mac1", **TINY}).status_code == 202


def test_a_kaggle_run_goes_up_trains_and_comes_home(places, fake_kaggle):
    """The follower end to end: bundle, upload, push, follow, fetch; with the Kaggle script itself
    run here on the bundle it uploaded (as Kaggle would run it)."""
    cloud.save_kaggle_credentials("me", "good-token")
    run = places["runs"] / "k2"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(TrainConfig(endgame_fraction=0, blocks=1, channels=16,
                                                            eval_opponents=("previous",), **TINY).to_json()))
    (run / "launch.json").write_text(json.dumps({"publish": True, "where": "kaggle"}))
    env = {**os.environ, "CHATURANGA_KAGGLE_CHECK": "0.05"}

    # 1. Upload and push, then stop following (as if the Mac restarted) before the session ends.
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "kaggle_run.py"), "--run", "k2", "--runs-dir",
                             str(places["runs"]), "--publish", str(places["published"])], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(1200):  # up to a minute: the upload is slow on a busy machine
        if (fake_kaggle / "run.py").exists():
            break
        import time
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    meta = json.loads((fake_kaggle / "kernel-metadata.json").read_text())
    assert meta["id"] == "me/chaturangzero-k2" and meta["is_private"] and meta["enable_gpu"]
    with tarfile.open(fake_kaggle / "bundle.tar.gz") as tar:
        names = tar.getnames()
    assert "code/chaturanga/training.py" in names and "run/config.json" in names

    # 2. "Kaggle" runs the script on the data, unpacked as Kaggle unpacks an uploaded archive.
    base = places["tmp"] / "kaggle-machine"
    data = base / "input" / "datasets" / "me" / "chaturangzero-k2-data"
    data.mkdir(parents=True)
    with tarfile.open(fake_kaggle / "bundle.tar.gz") as tar:
        tar.extractall(data)
    done = subprocess.run([sys.executable, str(fake_kaggle / "run.py")], cwd=str(base), capture_output=True, text=True,
                          env={**os.environ, "KAGGLE_BASE": str(base), "KAGGLE_WORKERS": "0"})
    assert done.returncode == 0, done.stdout + done.stderr
    (fake_kaggle / "working").symlink_to(base / "working")

    # 3. Following again picks up the same session (no second upload) and brings the run home.
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_run.py"), "--run", "k2", "--runs-dir",
                           str(places["runs"]), "--publish", str(places["published"])], env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Uploading" not in done.stdout
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "finished" and status["where"] == "kaggle"
    assert (run / "checkpoints" / "gen_0001.pt").exists() and (run / "kaggle.log").exists()
    assert json.loads((run / "cloud.json").read_text())["fetched"] is True
    assert places["published"].exists()
    detail = client.get("/api/training/runs/k2").json()
    assert detail["status"]["where"] == "kaggle" and [g["generation"] for g in detail["generations"]] == [1]


def test_a_session_stopped_on_kaggle_comes_home_and_is_not_started_again(places, fake_kaggle):
    """Stopped from its page on Kaggle (CANCEL_ACKNOWLEDGED): the follower notices, brings home what
    there is, and says so; one restarted before it did fetches that session instead of uploading anew."""
    cloud.save_kaggle_credentials("me", "good-token")
    run = places["runs"] / "k3"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(TrainConfig(**TINY).to_json()))
    (run / "launch.json").write_text(json.dumps({"publish": False, "where": "kaggle"}))
    env = {**os.environ, "CHATURANGA_KAGGLE_CHECK": "0.05", "FAKE_KAGGLE_END": "CANCEL_ACKNOWLEDGED"}
    follow = [sys.executable, str(ROOT / "scripts" / "kaggle_run.py"), "--run", "k3", "--runs-dir", str(places["runs"])]

    # Pushed, then the follower stops (the Mac restarted) while the session runs...
    proc = subprocess.Popen(follow, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(1200):
        if (fake_kaggle / "run.py").exists():
            break
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    # ...and the session is stopped on Kaggle's page.
    (fake_kaggle / "checks").write_text("3")
    pushes = (fake_kaggle / "kernel-metadata.json").stat().st_mtime

    done = subprocess.run(follow, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Uploading" not in done.stdout and (fake_kaggle / "kernel-metadata.json").stat().st_mtime == pushes
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "failed" and "stopped on Kaggle" in status["error"] and "lost" in status["error"]
    saved = json.loads((run / "cloud.json").read_text())
    assert saved["fetched"] is True and saved["outcome"] == "cancelacknowledged"


def test_deleting_a_waiting_colab_run_withdraws_its_request(places, monkeypatch):
    monkeypatch.setenv("CHATURANGA_DRIVE", str(places["drive"]))
    _no_spawn(monkeypatch)
    client.post("/api/training/runs", json={"name": "gone", "where": "colab", **TINY})
    client.post("/api/training/runs", json={"name": "kept", "where": "colab", **TINY})
    assert client.delete("/api/training/runs/gone").status_code == 200
    assert not (places["drive"] / "runs" / "gone").exists()
    assert cloud.take_request(places["drive"])["run"] == "kept"  # never the deleted one


@pytest.mark.parametrize("layout,workers", [("archive", 0), ("unpacked", 0), ("missing", 0), ("unpacked", 2)])
def test_the_kaggle_script_finds_its_code_however_kaggle_stores_it(tmp_path, layout, workers):
    """Kaggle unpacks uploaded archives; the script also takes the archive itself, and says what it
    found when the data is missing."""
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(TrainConfig(endgame_fraction=0, blocks=1, channels=16,
                                                            eval_opponents=("previous",), **TINY).to_json()))
    base, data = tmp_path / "kaggle", tmp_path / "kaggle" / "input" / "some-data"
    data.mkdir(parents=True)
    archive = cloud.bundle_run(run, tmp_path)
    if layout == "archive":
        (data / "bundle.tar.gz").write_bytes(archive.read_bytes())
    elif layout == "unpacked":
        with tarfile.open(archive) as tar:
            tar.extractall(data)
    else:
        (data / "other.csv").write_text("x")
    script = tmp_path / "run.py"
    script.write_text(cloud.KAGGLE_SCRIPT.format(run="r", hours=11.0))
    # With worker processes too, as on Kaggle's several cores (they import the script as they start).
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(tmp_path),
                          env={**os.environ, "KAGGLE_BASE": str(base), "KAGGLE_WORKERS": str(workers)}, timeout=300)
    if layout == "missing":
        assert done.returncode != 0 and "not in the attached data" in done.stderr and "other.csv" in done.stderr
    else:
        assert done.returncode == 0, done.stderr
        assert json.loads((base / "working" / "runs" / "r" / "status.json").read_text())["state"] == "finished"
