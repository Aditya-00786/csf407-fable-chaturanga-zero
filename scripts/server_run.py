"""Train a run on your own server over SSH, and follow it here: what the app's New run (or Resume)
does for Server.

    python scripts/server_run.py --run my-run      # a run folder made by the app (config.json, launch.json)

It uses the connection the app opened (Connect, with the server's password). It gets the server's
Python environment ready in the run's folder there (the first time takes a few minutes: PyTorch is
a few GB), copies the code (and the run, if the server does not have it yet), starts the training,
then every couple of seconds copies the run's status back into runs/<run>/status.json, where the
Training page reads it, and its metrics, games and networks every few more; pause requests made
here are passed on. The training does not depend on this process or the connection: if either
goes, it carries on, and starting this again (the app does, on Resume or Connect) follows it
instead of starting it twice. --follow-only never starts it.
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chaturanga import remote  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECK_EVERY = float(os.environ.get("CHATURANGA_SERVER_CHECK", 2))  # the run's status
PULL_EVERY = float(os.environ.get("CHATURANGA_SERVER_PULL", 15))  # its metrics, games and networks
RETRY_EVERY = float(os.environ.get("CHATURANGA_SERVER_RETRY", 10))  # while there is no connection
ACTIVE = {"starting", "selfplay", "training", "evaluating", "generation_done", "reporting"}

SETUP = {
    "env": "Making a Python environment in the run's folder",
    "torch": "Installing PyTorch there: a few GB, the first time only",
    "packages": "Installing the other packages",
    "ready": "Checking PyTorch and the GPU",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--generations", type=int, help="train up to this many generations")
    ap.add_argument("--follow-only", action="store_true", help="follow the run if it is training; never start it")
    args = ap.parse_args()
    run_dir = Path(args.runs_dir) / args.run
    status_file, notes_file = run_dir / "status.json", run_dir / "server.json"
    launch = json.loads((run_dir / "launch.json").read_text())
    host, folder = launch["host"], launch["folder"]
    workers = int(launch.get("workers") or 1)
    gpu_workers = int(launch.get("gpu_workers") or 0)
    try:
        status = json.loads(status_file.read_text())
    except (OSError, ValueError):
        status = {}
    try:
        notes = json.loads(notes_file.read_text())
    except (OSError, ValueError):
        notes = {}

    def write_status(new: dict) -> None:
        status.clear()
        status.update(new, run=args.run, where="server")
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        tmp.replace(status_file)

    def set_status(**changes) -> None:  # this process's own states, before the run reports its own
        write_status({**status, **changes, "updated": _now()})

    def note(**changes) -> None:  # how following goes: read by the API (server.json)
        notes.update(changes, follower_pid=os.getpid(), follower_host=socket.gethostname(), checked=_now())
        tmp = notes_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(notes, indent=2))
        tmp.replace(notes_file)

    def fail(message: str) -> None:
        set_status(state="failed", error=message, setup=None)
        note(connected=remote.connected(host), ended=_now())
        print(message, flush=True)
        sys.exit(1)

    note(connected=True, host=host, folder=folder, ended=None)
    pid = notes.get("remote_pid")
    try:
        # A run still training there (this process was restarted) is followed, not started again.
        there, alive = remote.poll(host, folder, args.run, pid) if remote.run_exists(host, folder, args.run) else (None, False)
        if there and there.get("pid") and not alive:
            pid = there["pid"]
            _, alive = remote.poll(host, folder, args.run, pid)
        if alive:
            print(f"{args.run} is training on {host}: following it.", flush=True)
        elif args.follow_only:
            if there:
                remote.pull_run(host, folder, args.run, run_dir)
                stopped = there.get("state") in ACTIVE  # it was training, and its process is gone
                write_status({**there, "state": "stopped"} if stopped else there)
            note(ended=_now())
            print(f"{args.run} is not training on {host}.", flush=True)
            return
        else:
            set_status(state="preparing", setup="Checking the server's Python environment", phase_started=_now(),
                       paused_in=None, paused_after=None, error=None)
            facts = remote.server_facts(host)
            check = None
            for step in remote.prepare(host, folder, facts.get("cuda")):
                if step.startswith("check="):
                    check = json.loads(step[6:])
                else:
                    print(f"  setup: {step}", flush=True)
                    set_status(setup=SETUP.get(step, step))
            note(torch=check)
            set_status(setup="Copying the code")
            remote.push_code(host, folder)
            fresh = not remote.run_exists(host, folder, args.run)
            set_status(setup="Copying the run" if fresh else "Copying the run's settings")
            remote.push_run(host, folder, args.run, run_dir, everything=fresh)
            (run_dir / "control.json").unlink(missing_ok=True)  # the run clears its own as it starts
            remote.send_control(host, folder, args.run, None)
            pid = remote.start_training(host, folder, args.run, workers, args.generations, gpu_workers)
            note(remote_pid=pid, started=_now(), hostname=facts.get("hostname"))
            gpu = check.get("gpu") if check else None
            on_gpu = f" ({min(gpu_workers, workers)} with the network on the GPU)" if gpu_workers and gpu else ""
            print(f"Started {args.run} on {host} (process {pid}): {workers} self-play processes{on_gpu}, "
                  f"training on {gpu or 'the CPU'}.", flush=True)
            set_status(state="starting", setup=None, phase_started=_now())
    except remote.NotConnected as exc:
        fail(str(exc))
    except remote.RemoteError as exc:
        fail(f"On the server: {exc}")

    # Follow: the status every few seconds, everything else every PULL_EVERY or when the run moves on.
    # The connection counts as back the moment the server answers; copying what the run made in the
    # meantime (minutes of it, after a long gap) is shown as syncing, not as still disconnected.
    pushed, last_pull, moved_on = None, 0.0, None

    def lost() -> None:
        if notes.get("connected"):
            print(f"Lost the connection to {host}; the run carries on there.", flush=True)
        note(connected=False, syncing=False)
        # Until the app connects again (a quick check here: nothing is sent), or a while has passed.
        waited = 0.0
        while waited < RETRY_EVERY and not remote.connected(host):
            time.sleep(1.0)
            waited += 1.0

    while True:
        time.sleep(CHECK_EVERY)
        try:
            control = run_dir / "control.json"
            want = control.read_text() if control.exists() else None
            if want != pushed:
                remote.send_control(host, folder, args.run, want)
                pushed = want
            there, alive = remote.poll(host, folder, args.run, pid)
            if not alive and there and there.get("pid") and there["pid"] != pid:
                # The run gives its own process id: follow that one if it is alive (an older status
                # left by an earlier session names a process long gone).
                _, theirs = remote.poll(host, folder, args.run, there["pid"])
                if theirs:
                    pid, alive = there["pid"], True
        except remote.NotConnected:
            lost()
            continue
        except remote.RemoteError as exc:  # a passing problem: try again
            print(f"Could not check the server: {exc}", flush=True)
            continue
        if not notes.get("connected"):
            print(f"Connected to {host} again.", flush=True)
        note(connected=True)
        reported = bool(there) and there.get("pid") == pid  # the status is this run's, not an earlier session's
        if reported:
            if there.get("state") not in ACTIVE:  # it has paused or ended, and cleared its own request there
                (run_dir / "control.json").unlink(missing_ok=True)
            write_status(there)
        mark = (there or {}).get("state"), (there or {}).get("generation")
        if not alive or mark != moved_on or time.time() - last_pull > PULL_EVERY:
            note(syncing=True)
            try:
                remote.pull_run(host, folder, args.run, run_dir)
                last_pull, moved_on = time.time(), mark
                note(syncing=False, synced=_now())
            except remote.NotConnected:
                lost()
                continue
            except remote.RemoteError as exc:
                print(f"Could not copy the run back: {exc}", flush=True)
                note(syncing=False)
                continue
        if not alive:
            break

    # The training process is gone: it paused, finished or failed (or was stopped some other way).
    if not reported:
        write_status({**status, "state": "failed", "setup": None, "updated": _now(),
                      "error": "The training stopped as it started on the server: the end of the run's train.log says why."})
    elif status.get("state") in ACTIVE:
        write_status({**status, "state": "stopped"})
    (run_dir / "control.json").unlink(missing_ok=True)
    note(ended=_now())
    print(f"{args.run} on {host}: {status.get('state')}.", flush=True)


if __name__ == "__main__":
    main()
