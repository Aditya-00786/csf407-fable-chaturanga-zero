"""Play a server run's strength check or rating ladder on its server, and follow it here: what the
run report's buttons do for a run that trains on your own server.

    python scripts/server_job.py --run my-run --job ladder --games 40 --every 10
    python scripts/server_job.py --run my-run --job strength --games 20
    python scripts/server_job.py --run my-run --job ladder --follow-only   # one playing there already

It uses the connection the app opened (Connect, with the server's password). It copies the code
there, starts scripts/report.py on the server (detached and at low priority, with the run's
processes and GPU settings), and every couple of seconds copies its progress file back
(report/benchmark.json or report/strength.json), where the run report reads it. The job does not
depend on this process or the connection: if either goes, it plays on, and the app follows it
again (--follow-only) once it can.
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
CHECK_EVERY = float(os.environ.get("CHATURANGA_SERVER_CHECK", 2))
RETRY_EVERY = float(os.environ.get("CHATURANGA_SERVER_RETRY", 10))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--job", required=True, choices=sorted(remote.REPORT_JOBS))
    ap.add_argument("--games", type=int)
    ap.add_argument("--every", type=int)
    ap.add_argument("--follow-only", action="store_true", help="follow the job playing there; never start one")
    args = ap.parse_args()
    run_dir = Path(args.runs_dir) / args.run
    file = remote.REPORT_JOBS[args.job][0]
    local = run_dir / file
    launch = json.loads((run_dir / "launch.json").read_text())
    host, folder = launch["host"], launch["folder"]
    try:
        mine = json.loads(local.read_text())
    except (OSError, ValueError):
        mine = {}

    def write(data: dict, **changes) -> None:
        """The job's progress here, with how following it goes (the API reads both)."""
        new = {**data, **changes, "where": "server", "server_host": host, "follower_pid": os.getpid(),
               "follower_host": socket.gethostname(), "checked": _now()}
        mine.clear()
        mine.update(new)
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".tmp")
        tmp.write_text(json.dumps(mine))
        tmp.replace(local)

    def fail(message: str) -> None:
        write(mine, state="failed", error=message, connected=remote.connected(host))
        print(message, flush=True)
        sys.exit(1)

    pid = mine.get("remote_pid")
    try:
        if not args.follow_only:
            remote.push_code(host, folder)
            if not remote.run_exists(host, folder, args.run):
                fail(f"The server has no run named {args.run!r}.")
            extra = ["--workers", str(int(launch.get("workers") or 1)),
                     "--gpu-workers", str(int(launch.get("gpu_workers") or 0))]
            extra += ["--games", str(args.games)] if args.games else []
            extra += ["--every", str(args.every)] if args.every and args.job == "ladder" else []
            pid = remote.start_report(host, folder, args.run, args.job, extra)
            write(mine, remote_pid=pid, connected=True)
            print(f"Started the {args.job} of {args.run} on {host} (process {pid}).", flush=True)
        elif not pid:
            fail("There is no job on the server to follow.")
    except remote.NotConnected as exc:
        fail(str(exc))
    except remote.RemoteError as exc:
        fail(f"On the server: {exc}")

    # Follow: its progress file every few seconds, until its process ends. Until the job has written
    # its own (naming its process), the file there is an earlier one's, and is not shown.
    there, ours = None, False
    while True:
        time.sleep(CHECK_EVERY)
        try:
            there, alive = remote.poll(host, folder, args.run, pid, file)
        except remote.NotConnected:
            if mine.get("connected") is not False:
                print(f"Lost the connection to {host}; the {args.job} plays on there.", flush=True)
                write(mine, connected=False)
            waited = 0.0
            while waited < RETRY_EVERY and not remote.connected(host):
                time.sleep(1.0)
                waited += 1.0
            continue
        except remote.RemoteError as exc:  # a passing problem: try again
            print(f"Could not check the server: {exc}", flush=True)
            continue
        ours = bool(there) and there.get("pid") == pid
        write(there if ours else mine, remote_pid=pid, connected=True)
        if not alive:
            break

    if not ours:
        tail = ""
        try:
            tail = remote.report_log(host, folder, args.run, args.job).strip().splitlines()[-1:]
        except remote.RemoteError:
            pass
        fail(f"The {args.job} stopped as it started on the server" + (f": {tail[0]}" if tail else "."))
    if mine.get("state") == "running":  # its process is gone without finishing
        write(mine, state="stopped")
    print(f"The {args.job} of {args.run} on {host}: {mine.get('state')}.", flush=True)


if __name__ == "__main__":
    main()
