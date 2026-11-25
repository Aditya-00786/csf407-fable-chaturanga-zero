"""Training on your own server over SSH (chaturanga/remote.py, scripts/server_run.py).

No test touches a real server: the server is a folder here, and ssh a small fake that signs in
with a password through ssh's askpass helper, as OpenSSH does, and then runs commands in that
folder. rsync is the real one, run through the fake ssh.
"""

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
if not (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")):
    pytest.skip("the API tests need httpx2: pip install -e '.[dev]'", allow_module_level=True)
if not shutil.which("rsync"):
    pytest.skip("these tests need rsync", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from chaturanga import remote  # noqa: E402
from chaturanga_server.app import app  # noqa: E402

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "correct horse battery"

FAKE_SSH = r'''#!{python}
"""A stand-in for ssh: a shared connection is a file; signing in asks SSH_ASKPASS for the password."""
import os, subprocess, sys
state, home = os.environ["FAKE_SSH_STATE"], os.environ["FAKE_SSH_HOME"]
args, opts, flags, i = sys.argv[1:], {{}}, set(), 0
while i < len(args):
    a = args[i]
    if a == "-o":
        k, _, v = args[i + 1].partition("="); opts[k] = v; i += 2
    elif a in ("-O", "-F", "-l", "-p"):
        opts[a] = args[i + 1]; i += 2
    elif a.startswith("-") and len(a) > 1:
        flags.update(a[1:]); i += 1
    else:
        break
host, command = args[i], " ".join(args[i + 1:])
marker = os.path.join(state, "open-" + host.replace("/", "_"))
with open(os.path.join(state, "calls.log"), "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
if "G" in flags:
    print("user tester"); print("hostname " + host + ".example"); sys.exit(0)
if "-O" in opts:
    if opts["-O"] == "check":
        sys.exit(0 if os.path.exists(marker) else 255)
    if opts["-O"] == "exit" and os.path.exists(marker):
        os.remove(marker)
    sys.exit(0)
if "M" in flags:
    if opts.get("BatchMode") == "yes":
        ok = os.environ.get("FAKE_SSH_KEY") == "1"
    else:
        assert os.environ.get("SSH_ASKPASS_REQUIRE") == "force"
        answer = subprocess.run([os.environ["SSH_ASKPASS"], "tester@" + host + "'s password: "],
                                capture_output=True, text=True, timeout=10).stdout
        declined = subprocess.run([os.environ["SSH_ASKPASS"], "Enter passphrase for key: "], capture_output=True).returncode
        ok = answer == os.environ["FAKE_SSH_PASSWORD"] + "\n" and declined != 0
    if not ok:
        sys.stderr.write("tester@" + host + ": Permission denied (publickey,password).\n"); sys.exit(255)
    open(marker, "w").close()
    sys.exit(0)
if not os.path.exists(marker):
    sys.stderr.write("tester@" + host + ": Permission denied (publickey,password).\n"); sys.exit(255)
slow = os.path.join(state, "slow-pull")
if "--sender" in command and os.path.exists(slow):  # copying a run back, made slow by a test
    import time
    time.sleep(float(open(slow).read()))
os.chdir(home)
os.environ["HOME"] = home
os.execv("/bin/sh", ["sh", "-c", command])
'''

# The server's Python environment, ready made: the tests' own Python (with torch), and pip calls noted.
FAKE_ENV_PYTHON = r'''#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then echo "$*" >> "$(dirname "$0")/../pip-calls.log"; exit 0; fi
exec {python} "$@"
'''


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A fake server (a home folder here), its ssh, local runs, and fast follower checks."""
    state, home, runs = tmp_path / "ssh-state", tmp_path / "server-home", tmp_path / "runs"
    for d in (state, home, runs):
        d.mkdir()
    ssh = _executable(tmp_path / "fake-ssh", FAKE_SSH.format(python=sys.executable))
    monkeypatch.setenv("CHATURANGA_SSH_TOOL", str(ssh))
    monkeypatch.setenv("FAKE_SSH_STATE", str(state))
    monkeypatch.setenv("FAKE_SSH_HOME", str(home))
    monkeypatch.setenv("FAKE_SSH_PASSWORD", PASSWORD)
    monkeypatch.setenv("CHATURANGA_RUNS", str(runs))
    monkeypatch.setenv("CHATURANGA_CHECKPOINT", str(tmp_path / "published" / "best.pt"))
    monkeypatch.setenv("CHATURANGA_SERVER_CHECK", "0.3")
    monkeypatch.setenv("CHATURANGA_SERVER_PULL", "1")
    monkeypatch.setenv("CHATURANGA_SERVER_RETRY", "0.5")
    return {"home": home, "runs": runs, "state": state, "folder": home / "proj" / "cz"}


def _ready_env(folder: Path, cuda=None) -> None:
    """Make the folder's environment as the first run would have, so no test installs anything."""
    env = folder / "env"
    (env / "bin").mkdir(parents=True)
    _executable(env / "bin" / "python", FAKE_ENV_PYTHON.format(python=sys.executable))
    torch, packages = remote.requirements()
    (env / "chaturanga-packages.json").write_text(remote.packages_note(remote.torch_index(cuda), torch, packages))


def _connect(**extra):
    return client.post("/api/training/server/connect",
                       json={"host": "lab", "folder": "~/proj/cz", "password": PASSWORD, "workers": 1, **extra})


def _wait(name, done, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        status = client.get("/api/training/runs").json()["runs"]
        mine = next((r for r in status if r["name"] == name), None)
        if mine and done(mine):
            return mine
        time.sleep(0.3)
    raise AssertionError(f"timed out; last status {mine}")


# ---------------------------------------------------------------------- connecting

def test_connecting_needs_the_right_password_and_keeps_it_nowhere(server):
    wrong = _connect(password="nope")
    assert wrong.status_code == 400 and "did not accept that password" in wrong.json()["detail"]
    assert client.get("/api/training/cloud").json()["server"]["connected"] is False

    ok = _connect()
    assert ok.status_code == 200, ok.text
    info = ok.json()["server"]
    assert info["connected"] is True and info["host"] == "lab" and info["workers"] == 1
    assert info["folder"] == "~/proj/cz" and info["folder_path"] == str(server["folder"])
    assert info["facts"]["home"] == str(server["home"]) and info["facts"]["cores"] >= 1
    # The password is in no file the app wrote, nor in any command it ran.
    for path in [*remote.app_dir().rglob("*"), server["state"] / "calls.log"]:
        if path.is_file():
            assert PASSWORD not in path.read_text(errors="replace"), path
    assert not list(remote.app_dir().rglob("password"))  # the pipe it went through is gone

    assert client.post("/api/training/server/disconnect").json()["server"]["connected"] is False


def test_a_server_that_takes_this_macs_key_needs_no_password(server, monkeypatch):
    assert client.post("/api/training/server/connect", json={"host": "lab"}).status_code == 400
    monkeypatch.setenv("FAKE_SSH_KEY", "1")
    ok = client.post("/api/training/server/connect", json={"host": "lab"})
    assert ok.status_code == 200 and ok.json()["server"]["folder"] == remote.DEFAULT_FOLDER


def test_hosts_and_folders_that_could_be_misread_are_refused(server):
    assert _connect(host="-oProxyCommand=evil").status_code == 400
    for folder in ("~/my runs", "~/a;rm -rf x", "relative/path"):
        assert _connect(folder=folder).status_code == 400, folder
    with pytest.raises(remote.RemoteError):
        remote.resolve_folder("~", "/home/me")  # the whole home folder


def test_hosts_from_the_ssh_config_are_offered(server, monkeypatch, tmp_path):
    config = tmp_path / "ssh_config"
    config.write_text("Host lab\n  HostName 10.0.0.5\n  User student\nHost *.internal !x\nHost linux vm\n")
    monkeypatch.setenv("CHATURANGA_SSH_CONFIG", str(config))
    hosts = client.get("/api/training/cloud").json()["server"]["hosts"]
    assert [h["name"] for h in hosts] == ["lab", "linux", "vm"]
    assert hosts[0] == {"name": "lab", "user": "tester", "hostname": "lab.example"}  # as `ssh -G` resolves it


def test_training_on_the_server_can_put_the_network_on_its_gpu(monkeypatch):
    scripts = []
    monkeypatch.setattr(remote, "run", lambda host, script, **kw: scripts.append(script) or "pid=42")
    assert remote.start_training("lab", "/srv/f", "big", workers=20, generations=40, gpu_workers=12) == 42
    assert "--workers 20" in scripts[0] and "--gpu-workers 12" in scripts[0] and "--generations 40" in scripts[0]
    remote.start_training("lab", "/srv/f", "small", workers=8)
    assert "--gpu-workers" not in scripts[1]


def test_a_server_run_with_gpu_processes_plays_more_games_per_process(server, monkeypatch):
    from chaturanga_server import training_api

    monkeypatch.setattr(training_api, "_spawn_server", lambda path, generations, status, follow_only=False: {})
    assert _connect().status_code == 200
    plain = client.post("/api/training/runs", json={"name": "cpu", "where": "server"})
    gpu = client.post("/api/training/runs", json={"name": "gpu", "where": "server", "workers": 10, "gpu_workers": 10,
                                                  "blocks": 10, "channels": 128, "simulations": 200})
    assert plain.status_code == gpu.status_code == 202
    config = {n: json.loads((server["runs"] / n / "config.json").read_text()) for n in ("cpu", "gpu")}
    launch = json.loads((server["runs"] / "gpu" / "launch.json").read_text())
    assert config["cpu"]["concurrent"] == 8 and config["gpu"]["concurrent"] == 32
    assert (launch["workers"], launch["gpu_workers"]) == (10, 10)
    assert (config["gpu"]["blocks"], config["gpu"]["channels"], config["gpu"]["simulations"]) == (10, 128, 200)


def test_the_apps_settings_for_a_run_never_travel_to_or_from_the_server(server):
    """Which network Play uses is chosen here; copying the run back must not undo the choice."""
    assert _connect().status_code == 200
    there, here = server["folder"] / "runs" / "s9", server["runs"] / "s9"
    there.mkdir(parents=True)
    here.mkdir(parents=True)
    (there / "launch.json").write_text(json.dumps({"play": "latest"}))  # stale, from an earlier start
    (there / "metrics.jsonl").write_text("{}\n")
    (here / "launch.json").write_text(json.dumps({"play": "best"}))
    (here / "config.json").write_text("{}")
    remote.pull_run("lab", str(server["folder"]), "s9", here)
    assert json.loads((here / "launch.json").read_text()) == {"play": "best"} and (here / "metrics.jsonl").exists()
    remote.push_run("lab", str(server["folder"]), "s9", here, everything=False)
    remote.push_run("lab", str(server["folder"]), "s9", here, everything=True)
    assert json.loads((there / "launch.json").read_text()) == {"play": "latest"}  # not sent either


def test_a_server_run_needs_a_connection(server):
    started = client.post("/api/training/runs", json={"name": "s1", "where": "server"})
    assert started.status_code == 409 and "Connect" in started.json()["detail"]


# ---------------------------------------------------------------------- the environment

def test_the_environment_is_made_once_and_kept_up_to_date(server):
    assert _connect().status_code == 200
    folder = str(server["folder"])
    _ready_env(server["folder"])
    steps = list(remote.prepare("lab", folder, None))
    assert steps[0] == "ready" and steps[-1].startswith("check=")  # nothing to install
    assert json.loads(steps[-1][6:])["torch"]
    assert not (server["folder"] / "env" / "pip-calls.log").exists() or "install" not in (server["folder"] / "env" / "pip-calls.log").read_text()

    # A newer requirement (or another GPU driver): installed again, torch from PyTorch's index for it.
    (server["folder"] / "env" / "chaturanga-packages.json").write_text("{}")
    steps = list(remote.prepare("lab", folder, "12.4"))
    assert steps[:3] == ["torch", "packages", "ready"]
    calls = (server["folder"] / "env" / "pip-calls.log").read_text()
    assert "--index-url https://download.pytorch.org/whl/cu124 torch" in calls and "numpy" in calls
    # Caches and temporary files stay in the folder.
    assert {"cache", "tmp", "code", "runs", "env"} <= {p.name for p in server["folder"].iterdir()}


# ---------------------------------------------------------------------- runs

def test_a_server_run_trains_there_and_comes_back_live(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    started = client.post("/api/training/runs", json={"name": "s1", "where": "server", "generations": 1,
                                                      "games_per_generation": 2, "simulations": 4, "eval_games": 2})
    assert started.status_code == 202, started.text
    assert started.json()["state"] == "preparing" and started.json()["live"] is True
    launch = json.loads((server["runs"] / "s1" / "launch.json").read_text())
    assert launch == {"play": "best", "where": "server", "host": "lab", "folder": str(server["folder"]), "workers": 1,
                      "gpu_workers": 0}

    done = _wait("s1", lambda r: not r["live"])
    assert done["state"] == "finished", (server["runs"] / "s1" / "server_run.log").read_text()
    local, there = server["runs"] / "s1", server["folder"] / "runs" / "s1"
    assert (there / "checkpoints" / "gen_0001.pt").exists()  # trained there...
    assert (local / "checkpoints" / "gen_0001.pt").exists() and (local / "metrics.jsonl").exists()  # ...and copied here
    assert (local / "games" / "gen_0001.jsonl.gz").exists() and "Done: 1 generations" in (local / "train.log").read_text()
    assert not list((local / "games").glob("*.partial.*"))  # merged into the generation's file, there and here
    assert (server["folder"] / "code" / "chaturanga" / "training.py").exists()
    detail = client.get("/api/training/runs/s1").json()
    assert detail["status"]["where"] == "server" and len(detail["generations"]) == 1
    # Its best network becomes the app's Trained opponent, as a cloud run's does.
    assert Path(os.environ["CHATURANGA_CHECKPOINT"]).exists()

    # Deleting it deletes the server's copy too, and needs the connection for that.
    client.post("/api/training/server/disconnect")
    assert client.delete("/api/training/runs/s1").status_code == 409
    _connect()
    assert client.delete("/api/training/runs/s1").status_code == 200
    assert not there.exists() and not local.exists()


def test_pausing_a_server_run_reaches_it_and_resume_carries_on_there(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    started = client.post("/api/training/runs", json={"name": "s2", "where": "server", "generations": 3,
                                                      "games_per_generation": 2, "simulations": 4, "eval_games": 0})
    assert started.status_code == 202, started.text
    _wait("s2", lambda r: r["state"] in ("selfplay", "training", "evaluating"))
    assert client.post("/api/training/runs/s2/pause", json={"when": "after_generation"}).status_code == 200
    paused = _wait("s2", lambda r: not r["live"])
    log = (server["runs"] / "s2" / "server_run.log").read_text()
    assert paused["state"] == "paused" and paused["generation"] < 3, log
    assert not (server["runs"] / "s2" / "control.json").exists()

    resumed = client.post("/api/training/runs/s2/resume", json={})
    assert resumed.status_code == 202 and resumed.json()["state"] == "preparing"
    done = _wait("s2", lambda r: not r["live"])
    assert done["state"] == "finished" and done["generation"] == 3, (server["runs"] / "s2" / "server_run.log").read_text()


def test_losing_the_connection_leaves_the_run_training_and_connect_follows_it_again(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    client.post("/api/training/runs", json={"name": "s3", "where": "server", "generations": 2,
                                            "games_per_generation": 4, "simulations": 8, "eval_games": 0})
    _wait("s3", lambda r: r["state"] == "selfplay")
    # The connection drops (the Mac slept, the network changed): the run carries on there.
    for marker in server["state"].glob("open-*"):
        marker.unlink()
    lost = _wait("s3", lambda r: r["server_connected"] is False or not r["live"])
    assert lost["live"] is True and lost["server_connected"] is False
    assert _connect().status_code == 200
    done = _wait("s3", lambda r: not r["live"])
    assert done["state"] == "finished" and done["generation"] == 2


def test_after_reconnecting_the_run_shows_connected_and_syncing_before_the_copy_is_done(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    client.post("/api/training/runs", json={"name": "s6", "where": "server", "generations": 2,
                                            "games_per_generation": 4, "simulations": 8, "eval_games": 0})
    _wait("s6", lambda r: r["state"] == "selfplay")
    for marker in server["state"].glob("open-*"):
        marker.unlink()
    _wait("s6", lambda r: r["server_connected"] is False)
    (server["state"] / "slow-pull").write_text("4")  # what the run made meanwhile takes a while to copy
    assert _connect().status_code == 200
    syncing = _wait("s6", lambda r: r["server_connected"] is True and r["server_syncing"] is True, timeout=30)
    assert syncing["live"] is True  # connected at once, still copying
    (server["state"] / "slow-pull").unlink()
    done = _wait("s6", lambda r: not r["live"])
    assert done["state"] == "finished" and done["server_syncing"] is False


def test_a_run_whose_follower_stopped_is_followed_again_on_connect(server):
    """This Mac restarted mid-run: Connect finds the run training there and follows it home."""
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    client.post("/api/training/runs", json={"name": "s4", "where": "server", "generations": 2,
                                            "games_per_generation": 4, "simulations": 8, "eval_games": 0})
    _wait("s4", lambda r: r["state"] == "selfplay")
    notes = json.loads((server["runs"] / "s4" / "server.json").read_text())
    os.kill(notes["follower_pid"], 9)
    stopped = _wait("s4", lambda r: not r["live"])
    assert stopped["state"] in ("selfplay", "training", "evaluating", "generation_done")  # last seen training
    assert _connect().status_code == 200
    done = _wait("s4", lambda r: r["live"] is False and r["state"] == "finished")
    assert done["generation"] == 2


def test_a_run_that_cannot_start_there_says_so(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    (server["folder"] / "env" / "bin" / "python").write_text("#!/bin/sh\n[ \"$1\" = -m ] && exit 0\n"
                                                            "case \"$*\" in *train.py*) echo 'no torch here' >&2; exit 1;; esac\n"
                                                            f"exec {sys.executable} \"$@\"\n")
    client.post("/api/training/runs", json={"name": "s5", "where": "server", "generations": 1,
                                            "games_per_generation": 2, "simulations": 4, "eval_games": 0})
    failed = _wait("s5", lambda r: not r["live"])
    assert failed["state"] == "failed" and "train.log" in failed["error"]
    assert "no torch here" in (server["runs"] / "s5" / "train.log").read_text()


# ---------------------------------------------------------------------- the run report's jobs there

def _wait_job(name, job, done, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        data = client.get(f"/api/training/runs/{name}/report").json()[job]
        if data and done(data):
            return data
        time.sleep(0.3)
    raise AssertionError(f"timed out; last {data}")


def test_a_server_runs_ladder_plays_on_the_server_and_comes_back(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    client.post("/api/training/runs", json={"name": "s7", "where": "server", "generations": 1,
                                            "games_per_generation": 2, "simulations": 2, "eval_games": 0})
    assert _wait("s7", lambda r: not r["live"])["state"] == "finished"
    plan = client.get("/api/training/runs/s7/report/ladder/plan", params={"games": 2, "every": 1}).json()
    assert plan["where"] == "server" and plan["entries"] == ["random", "greedy", "gen_0000", "gen_0001"]

    r = client.post("/api/training/runs/s7/report/ladder", json={"games": 2, "every": 1})
    assert r.status_code == 202 and r.json()["where"] == "server"
    # Training on the server waits for it.
    busy = client.post("/api/training/runs/s7/resume", json={"generations": 2})
    assert busy.status_code == 409 and "rating ladder of 's7'" in busy.json()["detail"]
    done = _wait_job("s7", "ladder", lambda d: d["state"] != "running")
    local, there = server["runs"] / "s7", server["folder"] / "runs" / "s7"
    assert done["state"] == "done", (local / "report" / "server_job.log").read_text()
    assert (there / "report" / "benchmark.json").exists() and "Rating ladder" in (there / "report" / "ladder.log").read_text()
    assert done["where"] == "server" and done["done"] == done["total"] == 6 * 2 + 3 * 40
    assert all(done["ratings"][e] is not None for e in plan["entries"])

    # The strength check of a server run plays there too.
    assert client.post("/api/training/runs/s7/report/strength", json={"games": 2}).status_code == 202
    strength = _wait_job("s7", "strength", lambda d: d["state"] != "running")
    assert strength["state"] == "done" and strength["where"] == "server"
    assert (there / "report" / "strength.json").exists()


def test_a_server_job_whose_follower_stopped_is_followed_again(server):
    assert _connect().status_code == 200
    _ready_env(server["folder"])
    client.post("/api/training/runs", json={"name": "s8", "where": "server", "generations": 1,
                                            "games_per_generation": 2, "simulations": 2, "eval_games": 0})
    _wait("s8", lambda r: not r["live"])
    client.post("/api/training/runs/s8/report/strength", json={"games": 4})
    playing = _wait_job("s8", "strength", lambda d: d.get("remote_pid") and d.get("total"))
    os.kill(playing["follower_pid"], 9)  # this Mac restarted, say
    time.sleep(0.5)
    done = _wait_job("s8", "strength", lambda d: d["state"] != "running")
    assert done["state"] == "done" and done["follower_pid"] != playing["follower_pid"]
