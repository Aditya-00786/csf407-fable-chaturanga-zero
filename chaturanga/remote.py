"""Training on your own server over SSH: what the app's New run does for Server.

Signing in. The server may ask for a password, which the app never keeps: Connect opens one SSH
connection with it (OpenSSH's connection sharing, ControlMaster), and every later command and copy
goes over that connection without signing in again. The password reaches ssh through a pipe, never
a file. The connection lasts until Disconnect, a network change or the server restarting; training
on the server carries on either way, and connecting again picks up where the app left off. A server
that takes this Mac's SSH key needs no password at all.

On the server everything lives in one folder (~/sopProject/chaturangzero unless you choose another):

    <folder>/env/        the Python environment the training runs in (made by the first run)
    <folder>/cache/      pip's downloads and other caches; <folder>/tmp/ for temporary files
    <folder>/code/       this repository's training code, copied before each start
    <folder>/runs/<run>  the runs, laid out as in runs/ here

scripts/server_run.py starts a run there and follows it: it copies the run back every few seconds,
so the Training page shows it live, and passes pause requests on.
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FOLDER = "~/sopProject/chaturangzero"
HOST = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,199}$")  # user@host or a name from ~/.ssh/config
FOLDER = re.compile(r"^(~|~/[A-Za-z0-9_./-]*|/[A-Za-z0-9_./-]*)$")  # no spaces or quotes: rsync and the shell take it as is
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


class RemoteError(RuntimeError):
    """An SSH command or copy failed; the message says why, in plain words."""


class NotConnected(RemoteError):
    """There is no open connection to the server: Connect (with its password) first."""


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------- settings on this Mac

def app_dir() -> Path:
    """Where the app keeps its own settings on this Mac (~/.chaturanga)."""
    return Path(os.environ.get("CHATURANGA_HOME", Path.home() / ".chaturanga"))


def settings() -> Optional[dict]:
    """The server new runs go to: host, folder (as typed and as a full path), cores to use, and
    what the server had when last connected. None until the first Connect."""
    try:
        data = json.loads((app_dir() / "server.json").read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("host") else None


def save_settings(**changes) -> dict:
    data = settings() or {}
    data.update(changes)
    folder = app_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "server.json.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(folder / "server.json")
    return data


def ssh_config_hosts() -> list[dict]:
    """The hosts named in ~/.ssh/config (not patterns), with the user and address each signs in as."""
    config = Path(os.environ.get("CHATURANGA_SSH_CONFIG", Path.home() / ".ssh" / "config"))
    try:
        lines = config.read_text().splitlines()
    except OSError:
        return []
    names = []
    for line in lines:
        words = line.split()
        if len(words) >= 2 and words[0].lower() == "host":
            names += [w for w in words[1:] if HOST.match(w) and w not in names]
    out = []
    for name in names:
        resolved = {"name": name, "user": None, "hostname": name}
        tool = ssh_tool()
        if tool:
            try:
                done = subprocess.run([tool, "-F", str(config), "-G", name], capture_output=True, text=True, timeout=5)
                for row in done.stdout.splitlines():
                    key, _, value = row.partition(" ")
                    if key in ("user", "hostname"):
                        resolved[key] = value
            except (OSError, subprocess.SubprocessError):
                pass
        out.append(resolved)
    return out


# ---------------------------------------------------------------------- the connection

def ssh_tool() -> Optional[str]:
    return os.environ.get("CHATURANGA_SSH_TOOL") or shutil.which("ssh")


def rsync_tool() -> Optional[str]:
    return os.environ.get("CHATURANGA_RSYNC_TOOL") or shutil.which("rsync")


def _sockets() -> Path:
    """The folder holding the shared connections (readable by you only). Kept short: a socket's
    path has a length limit."""
    folder = app_dir() / "ssh"
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    return folder


def _shared() -> list[str]:
    """ssh options that send a command over the open connection, and never ask for a password."""
    return ["-o", f"ControlPath={_sockets()}/%C", "-o", "ControlMaster=no", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15"]


def _check_host(host: str) -> str:
    if not HOST.match(host or ""):
        raise RemoteError(f"{host!r} is not a host name: use user@host, an address, or a name from ~/.ssh/config.")
    return host


def connected(host: str) -> bool:
    """Whether a connection to `host` is open (a quick local check: nothing is sent)."""
    tool = ssh_tool()
    if not tool or not HOST.match(host or ""):
        return False
    try:
        done = subprocess.run([tool, *_shared(), "-O", "check", host], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


# ssh runs this to ask for a password (SSH_ASKPASS); it answers password prompts only, with what
# the app writes into the pipe, and declines anything else (a key's passphrase, a one-time code).
ASKPASS = """#!/bin/sh
# Written by ChaturangZero (chaturanga/remote.py): hands ssh the password the app holds for a moment.
case "$1" in
  *assword*) exec cat "$CHATURANGA_ASKPASS_PIPE" ;;
esac
exit 1
"""


def _reader(pipe: str) -> Optional[int]:
    """The pipe opened for writing if ssh's helper has it open for reading, else None."""
    try:
        return os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:  # ENXIO: no reader
        return None


def _feed(pipe: str, password: str, proc: subprocess.Popen, times: int = 2) -> None:
    """Write the password into the pipe each time ssh's helper opens it (at most `times`), while ssh runs."""
    data = (password + "\n").encode()
    sent = 0
    while sent < times and proc.poll() is None:
        fd = _reader(pipe)
        if fd is None:
            time.sleep(0.05)
            continue
        try:
            os.set_blocking(fd, True)
            os.write(fd, data)
        except OSError:
            pass
        finally:
            os.close(fd)
        sent += 1
        # Once per prompt: wait for this helper to finish reading before serving another.
        while proc.poll() is None and (fd := _reader(pipe)) is not None:
            os.close(fd)
            time.sleep(0.05)


def _explain(err: str, host: str) -> str:
    text = (err or "").strip()
    low = text.lower()
    if "permission denied" in low:
        return "The server did not accept that password."
    if "host key verification failed" in low or "identification has changed" in low:
        return (f"{host} is not the machine this Mac knows by that name (its SSH host key has changed). "
                "Ask whoever runs it whether it was reinstalled.")
    if any(s in low for s in ("timed out", "no route", "could not resolve", "connection refused", "unreachable")):
        return f"Could not reach {host}. Is this Mac on the server's network, or its VPN?"
    return text.splitlines()[-1] if text else f"ssh could not connect to {host}."


def connect(host: str, password: Optional[str], timeout: float = 45) -> None:
    """Open the shared connection to `host`, signing in with `password` (None: with this Mac's SSH
    key). The password is passed to ssh through a pipe and not kept."""
    _check_host(host)
    if connected(host):
        return
    tool = ssh_tool()
    if not tool:
        raise RemoteError("This Mac has no ssh command.")
    sockets = _sockets()
    cmd = [tool, "-M", "-N", "-f", "-o", f"ControlPath={sockets}/%C", "-o", "ControlPersist=yes",
           "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
           "-o", "ConnectTimeout=15"]
    if not password:
        proc = subprocess.run([*cmd, "-o", "BatchMode=yes", host], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RemoteError(_explain(proc.stderr, host) if "permission denied" not in proc.stderr.lower()
                              else f"{host} asks for a password.")
        return
    helper = sockets / "askpass.sh"
    helper.write_text(ASKPASS)
    os.chmod(helper, 0o700)
    with tempfile.TemporaryDirectory(dir=sockets) as private:
        pipe = os.path.join(private, "password")
        os.mkfifo(pipe, 0o600)
        env = {**os.environ, "SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force",
               "CHATURANGA_ASKPASS_PIPE": pipe}
        proc = subprocess.Popen([*cmd, "-o", "NumberOfPasswordPrompts=1", host], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
        threading.Thread(target=_feed, args=(pipe, password, proc), name="askpass", daemon=True).start()
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # ssh and its helper; no connection was made
            proc.communicate()
            raise RemoteError(f"{host} did not answer in time.")
    if proc.returncode != 0 or not connected(host):
        raise RemoteError(_explain(err, host))


def disconnect(host: str) -> None:
    tool = ssh_tool()
    if tool and HOST.match(host or ""):
        subprocess.run([tool, *_shared(), "-O", "exit", host], capture_output=True, text=True, timeout=10)


def run(host: str, script: str, timeout: float = 60, input: Optional[str] = None, check: bool = True) -> str:
    """Run a POSIX shell script on the server over the open connection; its output."""
    tool = ssh_tool()
    if not tool:
        raise RemoteError("This Mac has no ssh command.")
    try:
        done = subprocess.run([tool, *_shared(), _check_host(host), "sh -c " + shlex.quote(script)],
                              capture_output=True, text=True, timeout=timeout, input=input,
                              stdin=None if input is not None else subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise RemoteError(f"{host} did not answer in time.") from exc
    if done.returncode == 255:  # ssh itself failed: the connection is gone
        raise NotConnected(f"Not connected to {host}: connect with its password.")
    if check and done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        raise RemoteError(lines[-1] if lines else f"a command on {host} failed")
    return done.stdout


def stream(host: str, script: str, timeout: float = 3600) -> Iterator[str]:
    """Run a script on the server, yielding its output lines as they come."""
    tool = ssh_tool()
    if not tool:
        raise RemoteError("This Mac has no ssh command.")
    proc = subprocess.Popen([tool, *_shared(), _check_host(host), "sh -c " + shlex.quote(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    timer = threading.Timer(timeout, proc.kill)
    timer.start()
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
        err = proc.stderr.read()
        proc.wait()
    finally:
        timer.cancel()
    if proc.returncode == 255:
        raise NotConnected(f"Not connected to {host}: connect with its password.")
    if proc.returncode != 0:
        lines = err.strip().splitlines()
        raise RemoteError(lines[-1] if lines else f"a command on {host} failed")


def rsync(host: str, sources: list[str], dest: str, *options: str, timeout: float = 1800) -> None:
    """Copy with rsync over the open connection (a remote path is host:/path)."""
    tool, ssh = rsync_tool(), ssh_tool()
    if not tool or not ssh:
        raise RemoteError("This Mac needs ssh and rsync to copy to the server.")
    cmd = [tool, "-az", "--timeout=120", "-e", " ".join([ssh, *_shared()]), *options, *sources, dest]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise RemoteError("Copying to or from the server took too long.") from exc
    # 24: some files vanished while copying (a run's work files come and go); the rest arrived.
    if done.returncode not in (0, 24):
        out = (done.stderr or done.stdout).strip()
        if done.returncode == 255 or "connection unexpectedly closed" in out or "Permission denied" in out:
            raise NotConnected(f"Not connected to {host}: connect with its password.")
        raise RemoteError(out.splitlines()[-1] if out else "rsync failed")


# ---------------------------------------------------------------------- the server

FACTS = r"""echo "home=$HOME"
echo "hostname=$(hostname)"
echo "cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1 | sed 's/^/cpu=/'
free -g 2>/dev/null | awk '/^Mem:/ {print "memory_gb=" $2}'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/gpu=/'
nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/cuda=\1/p' | head -1
python3 -c 'import sys; print("python=%d.%d.%d" % sys.version_info[:3])' 2>/dev/null
command -v rsync >/dev/null 2>&1 && echo "rsync=yes"
"""


def server_facts(host: str) -> dict:
    """What the server has: cores, memory, GPU, Python, rsync, and its home folder."""
    facts: dict = {}
    for line in run(host, FACTS).splitlines():
        key, sep, value = line.partition("=")
        if sep and value.strip():
            facts[key.strip()] = value.strip()
    for key in ("cores", "memory_gb"):
        if key in facts:
            try:
                facts[key] = int(facts[key])
            except ValueError:
                del facts[key]
    facts["rsync"] = facts.get("rsync") == "yes"
    return facts


def default_workers(cores: Optional[int]) -> int:
    """Self-play processes on a server: all cores but a few, left for the training itself and for
    anyone else using the machine."""
    return max(1, (cores or 2) - 4) if (cores or 0) > 8 else max(1, (cores or 2) - 1)


def resolve_folder(folder: str, home: str) -> str:
    """The folder as a full path on the server (~ is its home folder)."""
    if not FOLDER.match(folder or ""):
        raise RemoteError("The folder is a path such as ~/sopProject/chaturangzero, without spaces or quotes.")
    if folder == "~" or folder.startswith("~/"):
        folder = home.rstrip("/") + folder[1:]
    folder = os.path.normpath(folder)
    if folder in ("/", home.rstrip("/")):
        raise RemoteError("Use a folder of its own (inside your home folder, say), not the whole home folder.")
    return folder


def requirements() -> tuple[str, list[str]]:
    """What the training needs, from pyproject.toml: torch (installed apart, from PyTorch's own
    index) and the rest (plus matplotlib, for a run that makes its own report)."""
    text = (ROOT / "pyproject.toml").read_text()
    try:
        import tomllib

        data = tomllib.loads(text)["project"]
        packages = list(data["dependencies"]) + list(data.get("optional-dependencies", {}).get("train", []))
    except ImportError:  # Python 3.10
        packages = re.findall(r'"([A-Za-z0-9_.-]+[<>=!~][^"]*)"', text.split("[project.optional-dependencies]")[0])
        packages += re.findall(r'"(matplotlib[^"]*)"', text)
    torch = next((p for p in packages if re.match(r"torch\b", p)), "torch")
    return torch, [p for p in packages if p != torch]


# PyTorch's CUDA builds, newest first: the server gets the newest its NVIDIA driver runs (the driver's
# "CUDA Version" in nvidia-smi), or the much smaller CPU build without one.
CUDA_BUILDS = [((13, 0), "cu130"), ((12, 9), "cu129"), ((12, 8), "cu128"), ((12, 6), "cu126"),
               ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]


def torch_index(cuda: Optional[str]) -> str:
    if cuda:
        try:
            version = tuple(int(x) for x in cuda.split(".")[:2])
        except ValueError:
            version = ()
        for least, tag in CUDA_BUILDS:
            if version >= least:
                return f"https://download.pytorch.org/whl/{tag}"
    return "https://download.pytorch.org/whl/cpu"


def _env_exports() -> str:
    """Caches and temporary files go in the folder too, not the home folder."""
    return ('export PIP_CACHE_DIR="$F/cache/pip" XDG_CACHE_HOME="$F/cache" TMPDIR="$F/tmp" '
            'MPLCONFIGDIR="$F/cache/matplotlib" TORCH_HOME="$F/cache/torch" PIP_DISABLE_PIP_VERSION_CHECK=1')


def packages_note(index: str, torch: str, packages: list[str]) -> str:
    """What an environment was made with (kept in env/chaturanga-packages.json): when the packages
    asked for change, the environment is brought up to date."""
    return json.dumps({"index": index, "torch": torch, "packages": packages}, sort_keys=True)


def setup_script(folder: str, index: str, torch: str, packages: list[str]) -> str:
    """The script that gets the folder's Python environment ready (quick when it is). It prints
    step=<name> as it goes: env, torch, packages, then ready, with check=<json> at the end."""
    q = shlex.quote
    want = packages_note(index, torch, packages)
    return f"""set -e
F={q(folder)}
mkdir -p "$F/cache" "$F/tmp" "$F/code" "$F/runs"
{_env_exports()}
WANT={q(want)}
if ! "$F/env/bin/python" -m pip --version > /dev/null 2>&1 || [ "$(cat "$F/env/chaturanga-packages.json" 2>/dev/null)" != "$WANT" ]; then
  if ! "$F/env/bin/python" -m pip --version > /dev/null 2>&1; then  # none yet, or one left half made
    echo "step=env"
    rm -rf "$F/env"
    if ! python3 -m venv "$F/env" > "$F/tmp/venv.log" 2>&1; then
      # Ubuntu without its python3-venv package: an environment without pip, then pip from PyPA's own installer.
      rm -rf "$F/env"
      python3 -m venv --without-pip "$F/env"
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$F/tmp/get-pip.py" || wget -qO "$F/tmp/get-pip.py" https://bootstrap.pypa.io/get-pip.py
      "$F/env/bin/python" "$F/tmp/get-pip.py" -q
    fi
  fi
  "$F/env/bin/python" -m pip install -q --upgrade pip
  echo "step=torch"
  "$F/env/bin/python" -m pip install -q --index-url {q(index)} {q(torch)}
  echo "step=packages"
  "$F/env/bin/python" -m pip install -q {" ".join(q(p) for p in packages)}
  printf '%s' "$WANT" > "$F/env/chaturanga-packages.json"
fi
echo "step=ready"
"$F/env/bin/python" -c 'import json, torch; c = torch.cuda.is_available(); print("check=" + json.dumps({{"torch": torch.__version__, "cuda": c, "gpu": torch.cuda.get_device_name(0) if c else None}}))'
"""


def prepare(host: str, folder: str, cuda: Optional[str]) -> Iterator[str]:
    """Get the server's environment ready, yielding each step's name as it starts (see setup_script);
    the last item is the check, as JSON."""
    torch, packages = requirements()
    for line in stream(host, setup_script(folder, torch_index(cuda), torch, packages)):
        if line.startswith("step="):
            yield line[5:]
        elif line.startswith("check="):
            yield line


def push_code(host: str, folder: str) -> None:
    """Copy the training code into <folder>/code (only what changed)."""
    rsync(host, [str(ROOT / "chaturanga"), str(ROOT / "scripts"), str(ROOT / "pyproject.toml")],
          f"{host}:{folder}/code/", "--delete", "--exclude=__pycache__", "--exclude=*.pyc")


# Files that belong to one side only: the run's live status and pause requests (passed on by the
# follower), the follower's own notes and log here, and the app's settings for the run (launch.json:
# where it trains, and which of its networks Play uses), which the server neither needs nor changes.
LOCAL_ONLY = ("/status.json", "/control.json", "/server.json", "/server_run.log", "/launch.json", "*.tmp")


def run_path(folder: str, name: str) -> str:
    if not NAME.match(name):
        raise RemoteError(f"{name!r} is not a run name")
    return f"{folder}/runs/{name}"


def run_exists(host: str, folder: str, name: str) -> bool:
    return run(host, f'test -f {shlex.quote(run_path(folder, name))}/config.json && echo yes || true').strip() == "yes"


def push_run(host: str, folder: str, name: str, local: Path, everything: bool) -> None:
    """Copy the run to the server: all of it (a new run, or one that trained elsewhere so far), or
    just its training settings (the server has the rest, and may have changed it since)."""
    remote = run_path(folder, name)
    run(host, f"mkdir -p {shlex.quote(remote)}")
    if everything:
        rsync(host, [f"{local}/"], f"{host}:{remote}/", *(f"--exclude={p}" for p in LOCAL_ONLY))
    else:
        if (local / "config.json").exists():
            rsync(host, [str(local / "config.json")], f"{host}:{remote}/")


def pull_run(host: str, folder: str, name: str, local: Path) -> None:
    """Copy what the run has made on the server into the run's folder here (its metrics, games,
    networks and log). Nothing here is deleted: a report made here stays."""
    rsync(host, [f"{host}:{run_path(folder, name)}/"], f"{local}/", *(f"--exclude={p}" for p in LOCAL_ONLY))
    # A generation's games in progress (gen_NNNN.partial.<pid>.jsonl.gz) are merged into its file
    # when it is done, and removed there: here too, once that file has arrived.
    for partial in (local / "games").glob("gen_*.partial.*.jsonl.gz"):
        if (partial.parent / f"{partial.name.split('.')[0]}.jsonl.gz").exists():
            partial.unlink(missing_ok=True)


def start_training(host: str, folder: str, name: str, workers: int, generations: Optional[int] = None,
                   gpu_workers: int = 0) -> int:
    """Start scripts/train.py on the server, detached from the connection (it trains on if the
    connection drops) and at low priority (others using the machine come first); its process id.
    `gpu_workers` of the self-play processes run the network on the server's GPU."""
    q = shlex.quote
    extra = f" --generations {int(generations)}" if generations else ""
    extra += f" --gpu-workers {int(gpu_workers)}" if gpu_workers else ""
    out = run(host, f"""F={q(folder)}
{_env_exports()}
cd "$F/code"
D={q(run_path(folder, name))}
S=$(command -v setsid || true)  # its own session where there is setsid (Linux), so nothing ties it to this one
$S nohup nice -n 10 "$F/env/bin/python" -u scripts/train.py --run {q(name)} --runs-dir "$F/runs" \\
  --workers {int(workers)} --publish ''{extra} >> "$D/train.log" 2>&1 < /dev/null &
echo "pid=$!"
""")
    match = re.search(r"pid=(\d+)", out)
    if not match:
        raise RemoteError("The training did not start on the server.")
    return int(match.group(1))


# Report jobs a server run plays there, as scripts/report.py does them: (progress file, flag).
REPORT_JOBS = {"strength": ("report/strength.json", "--strength-only"),
               "ladder": ("report/benchmark.json", "--ladder-only")}


def start_report(host: str, folder: str, name: str, job: str, args: list[str]) -> int:
    """Start one of the run report's jobs (the strength check or the rating ladder) on the server,
    detached and at low priority as the training is; its process id. Its log is report/<job>.log."""
    q = shlex.quote
    flag, extra = REPORT_JOBS[job][1], " ".join(q(a) for a in args)
    out = run(host, f"""F={q(folder)}
{_env_exports()}
cd "$F/code"
D={q(run_path(folder, name))}
mkdir -p "$D/report"
S=$(command -v setsid || true)
$S nohup nice -n 10 "$F/env/bin/python" -u scripts/report.py --run {q(name)} --runs-dir "$F/runs" {flag} \\
  {extra} >> "$D/report/{job}.log" 2>&1 < /dev/null &
echo "pid=$!"
""")
    match = re.search(r"pid=(\d+)", out)
    if not match:
        raise RemoteError(f"The {job} did not start on the server.")
    return int(match.group(1))


def report_log(host: str, folder: str, name: str, job: str, lines: int = 20) -> str:
    d = shlex.quote(run_path(folder, name))
    return run(host, f"tail -n {int(lines)} {d}/report/{job}.log 2>/dev/null || true")


def poll(host: str, folder: str, name: str, pid: Optional[int], file: str = "status.json") -> tuple[Optional[dict], bool]:
    """The run's status on the server (or another of its JSON files), and whether the process
    `pid` (its training, or a report job) is alive."""
    if not re.fullmatch(r"[\w/.-]+\.json", file) or ".." in file:
        raise RemoteError(f"{file!r} is not a run's file")
    d = shlex.quote(run_path(folder, name))
    check = f"kill -0 {int(pid)} 2>/dev/null && echo alive=yes || echo alive=no" if pid else "echo alive=no"
    out = run(host, f'cat {d}/{file} 2>/dev/null; echo; echo "--"; {check}')
    body, _, tail = out.rpartition("--")
    try:
        status = json.loads(body.strip()) if body.strip() else None
    except ValueError:
        status = None  # caught mid-write: next time
    return status, "alive=yes" in tail


def send_control(host: str, folder: str, name: str, request: Optional[str]) -> None:
    """Pass a pause request (control.json's contents) on to the run there; None withdraws it."""
    d = shlex.quote(run_path(folder, name))
    if request is None:
        run(host, f"rm -f {d}/control.json")
    else:
        run(host, f"cat > {d}/control.json.tmp && mv {d}/control.json.tmp {d}/control.json", input=request)


def delete_run(host: str, folder: str, name: str) -> None:
    run(host, f"rm -rf -- {shlex.quote(run_path(folder, name))}", timeout=300)
