"""Training in the cloud, started from the app: Colab (through Google Drive) and Kaggle (through its API).

Colab. Google offers no way to start a Colab notebook from another program, so the app does what it
can: it writes the run's folder and a request into the ChaturangZero folder of Google Drive and
opens the notebook in the browser, where Run all takes the request (`take_request`) and trains
into Drive. With Google Drive for desktop, the same folder appears on this Mac, so the app follows
the run, pauses it (control.json syncs back) and reads its results like any other.

Kaggle. Kaggle's command-line tool can upload data, start a notebook on a GPU, report its state
and download its output, so scripts/kaggle_run.py does all of that on its own: the run's code
(and its saved state, when resuming) goes up as a private dataset, a private script trains with
it, and the finished run comes back into runs/. Kaggle sessions end after 12 hours; the script
pauses the run after the generation in progress at 11, so it can be resumed in a new session.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Aditya-00786/csf407-fable-chaturanga-zero"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- Colab, through Google Drive

def drive_folder() -> Optional[Path]:
    """The ChaturangZero folder in Google Drive as Google Drive for desktop shows it on this Mac
    (CHATURANGA_DRIVE names it directly), or None if Drive is not synced here."""
    if "CHATURANGA_DRIVE" in os.environ:  # set, even to nothing ("": no Drive here), it decides
        named = os.environ["CHATURANGA_DRIVE"]
        return Path(named) if named else None
    for account in sorted((Path.home() / "Library" / "CloudStorage").glob("GoogleDrive-*")):
        my_drive = account / "My Drive"
        if my_drive.is_dir():
            return my_drive / "ChaturangZero"
    return None


def repo_slug() -> str:
    """owner/name of the GitHub repository, from the origin remote."""
    try:
        url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        url = ""
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return match.group(1) if match else DEFAULT_REPO


def colab_url(branch: Optional[str] = None) -> str:
    """The training notebook, opened in Colab straight from GitHub (Colab asks to connect GitHub
    once, for the private repository)."""
    branch = branch or os.environ.get("CHATURANGA_COLAB_BRANCH", "main")
    return f"https://colab.research.google.com/github/{repo_slug()}/blob/{branch}/notebooks/train_colab.ipynb"


def request_colab_run(drive: Path, name: str) -> None:
    """Ask the next Colab session to train the run `name` (its folder is already in Drive)."""
    folder = drive / "requests"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.json.tmp"
    tmp.write_text(json.dumps({"run": name, "requested": _now()}))
    tmp.replace(folder / f"{name}.json")


def take_request(drive) -> Optional[dict]:
    """In the Colab notebook: the oldest waiting request, marked as taken so no other session
    picks it up; None if there is none."""
    folder = Path(drive) / "requests"
    waiting = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    for path in waiting:
        try:
            request = json.loads(path.read_text())
            path.replace(path.with_suffix(".taken"))
        except (OSError, ValueError):
            continue
        if (Path(drive) / "runs" / request.get("run", "") / "config.json").is_file():
            return request  # (a request for a run deleted since is skipped)
    return None


# ---------------------------------------------------------------------- Kaggle

class KaggleError(RuntimeError):
    """The Kaggle tool failed; the message says why, in Kaggle's words."""


def kaggle_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))


def kaggle_credentials() -> Optional[dict]:
    """How to sign in to Kaggle, or None: an API token saved by the app (access_token, with the
    username in chaturangzero.json), or the older username and key (kaggle.json or the
    environment)."""
    folder = kaggle_dir()
    token = folder / "access_token"
    try:
        username = json.loads((folder / "chaturangzero.json").read_text()).get("username")
    except (OSError, ValueError):
        username = None
    if token.exists() and username:
        return {"username": username, "token_file": str(token)}
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}
    try:
        data = json.loads((folder / "kaggle.json").read_text())
        return {"username": data["username"], "key": data["key"]} if data.get("username") and data.get("key") else None
    except (OSError, ValueError, KeyError):
        return None


def _kaggle_env(credentials: dict) -> dict:
    """The environment that signs the Kaggle tool in with these credentials."""
    env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")}
    if credentials.get("token_file") or credentials.get("token"):
        env["KAGGLE_API_TOKEN"] = credentials.get("token") or credentials["token_file"]  # a file path works too
    else:
        env.update(KAGGLE_USERNAME=credentials["username"], KAGGLE_KEY=credentials["key"])
    return env


def save_kaggle_credentials(username: str, token: str) -> None:
    """Keep the API token where the Kaggle tool looks for it (readable by this user only), and the
    username, which names the run's private files on Kaggle."""
    folder = kaggle_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "access_token.tmp"
    tmp.write_text(token)
    os.chmod(tmp, 0o600)
    tmp.replace(folder / "access_token")
    (folder / "chaturangzero.json").write_text(json.dumps({"username": username}))


def kaggle_tool() -> Optional[str]:
    """The kaggle command: next to this Python (installed with pip install -e '.[cloud]'), or on PATH."""
    if os.environ.get("CHATURANGA_KAGGLE_TOOL"):
        return os.environ["CHATURANGA_KAGGLE_TOOL"]
    beside = Path(sys.executable).parent / "kaggle"
    return str(beside) if beside.exists() else shutil.which("kaggle")


def kaggle(*args: str, timeout: float = 600, credentials: Optional[dict] = None) -> str:
    """Run the Kaggle tool, signed in with `credentials` (default: the saved ones)."""
    tool = kaggle_tool()
    if not tool:
        raise KaggleError("The Kaggle tool is not installed: pip install -e '.[cloud]'")
    credentials = credentials or kaggle_credentials()
    env = _kaggle_env(credentials) if credentials else None
    done = subprocess.run([tool, *args], capture_output=True, text=True, timeout=timeout, env=env)
    out = (done.stdout + done.stderr).strip()
    if done.returncode != 0 or "401" in out or "403" in out or "Authentication required" in out:
        raise KaggleError(out.splitlines()[-1] if out else f"kaggle {args[0]} failed")
    return out


def slug(name: str) -> str:
    """A Kaggle-safe slug for a run name (lower case, hyphens)."""
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:40] or "run"


def kernel_id(username: str, name: str) -> str:
    return f"{username}/chaturangzero-{slug(name)}"


def kernel_url(username: str, name: str) -> str:
    return f"https://www.kaggle.com/code/{kernel_id(username, name)}"


# The script Kaggle runs: unpack the code (and the run so far), train, leave the run in /kaggle/working.
KAGGLE_SCRIPT = '''"""ChaturangZero training on Kaggle, prepared by scripts/kaggle_run.py."""
import os, shutil, sys, tarfile, threading
RUN = {run!r}
HOURS = {hours!r}
BASE = os.environ.get("KAGGLE_BASE", "/kaggle")  # set only by the tests
bundle = next(os.path.join(d, f) for d, _, fs in os.walk(f"{{BASE}}/input") for f in fs if f == "bundle.tar.gz")
with tarfile.open(bundle) as tar:
    tar.extractall(f"{{BASE}}/tmp")
sys.path.insert(0, f"{{BASE}}/tmp/code")
run_dir = f"{{BASE}}/working/runs/{{RUN}}"
if os.path.isdir(f"{{BASE}}/tmp/run"):  # the run's settings, and its progress when resuming
    shutil.copytree(f"{{BASE}}/tmp/run", run_dir, dirs_exist_ok=True)
from chaturanga.training import TrainingRun, request_pause
# Kaggle ends a session at 12 hours: pause after the generation in progress before then.
timer = threading.Timer(HOURS * 3600, lambda: request_pause(run_dir, "after_generation"))
timer.daemon = True
timer.start()
workers = int(os.environ.get("KAGGLE_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
TrainingRun(run_dir, workers=workers).run()
'''


def bundle_run(run_dir: Path, out: Path) -> Path:
    """The code and the run's saved state, as one archive for a Kaggle dataset."""
    archive = out / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(ROOT / "chaturanga", "code/chaturanga", filter=lambda t: None if "__pycache__" in t.name else t)
        for item in ("config.json", "metrics.jsonl", "checkpoints", "games", "launch.json"):
            if (run_dir / item).exists():
                tar.add(run_dir / item, f"run/{item}")
    return archive


def push_run(run_dir: Path, username: str, hours: float = 11.0) -> str:
    """Upload the code and run as a private dataset and start the training script on a Kaggle GPU.
    Returns the script's id."""
    name = run_dir.name
    dataset = f"{username}/chaturangzero-{slug(name)}-data"
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, kernel_dir = Path(tmp) / "data", Path(tmp) / "kernel"
        data_dir.mkdir()
        kernel_dir.mkdir()
        bundle_run(run_dir, data_dir)
        (data_dir / "dataset-metadata.json").write_text(json.dumps(
            {"title": f"chaturangzero-{slug(name)}-data", "id": dataset, "licenses": [{"name": "other"}]}))
        try:
            kaggle("datasets", "status", dataset, timeout=60)
            kaggle("datasets", "version", "-p", str(data_dir), "-m", f"run {name} at {_now()}", "-q")
        except KaggleError:
            kaggle("datasets", "create", "-p", str(data_dir), "-q")  # private unless --public
        for _ in range(60):  # wait until Kaggle has processed the upload
            if "ready" in kaggle("datasets", "status", dataset, timeout=60).lower():
                break
            time.sleep(10)
        script = kernel_dir / "run.py"
        script.write_text(KAGGLE_SCRIPT.format(run=name, hours=hours))
        kid = kernel_id(username, name)
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
            "dataset_sources": [dataset], "competition_sources": [], "kernel_sources": []}))
        kaggle("kernels", "push", "-p", str(kernel_dir))
    return kid


def kernel_state(kid: str) -> str:
    """queued, running, complete, error or cancel..., as Kaggle reports it."""
    out = kaggle("kernels", "status", kid, timeout=60)
    match = re.search(r'status "?([A-Za-z_.]+)', out)
    word = match.group(1) if match else out.split()[-1]
    return word.rsplit(".", 1)[-1].strip('"').lower()  # "KernelWorkerStatus.COMPLETE" or "complete"


def fetch_output(kid: str, run_dir: Path) -> Optional[Path]:
    """Download the script's output and put the run back in run_dir. Returns the session's log file."""
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", kid, "-p", tmp, "-o", timeout=1800)
        got = Path(tmp) / "runs" / run_dir.name
        if got.is_dir():
            for item in got.iterdir():
                target = run_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
        logs = list(Path(tmp).glob("*.log"))
        if logs:
            keep = run_dir / "kaggle.log"
            shutil.copy2(logs[0], keep)
            return keep
    return None
