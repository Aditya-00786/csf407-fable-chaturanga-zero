"""Train a run on Kaggle and bring it home: what the app's New run (or Resume) does for Kaggle.

    python scripts/kaggle_run.py --run my-run            # a run folder made by the app (config.json)

It uploads the code and the run so far as a private Kaggle dataset, starts a private script on a
Kaggle GPU, checks it every minute, and when the session ends downloads the run into runs/<run>.
Its progress is kept in runs/<run>/status.json (where the Training page reads it) and the Kaggle
script's id in runs/<run>/cloud.json. If this process stops (the Mac restarts), starting it again
follows the same Kaggle session instead of starting another, or brings it home if it has ended
(finished, failed or stopped on Kaggle's page); a new session starts only once the last one is home.
"""

import argparse
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chaturanga.cloud import (KaggleError, fetch_output, kaggle_credentials, kernel_state,  # noqa: E402
                              kernel_url, push_run)

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECK_EVERY = float(os.environ.get("CHATURANGA_KAGGLE_CHECK", 60))
ENDED = ("complete", "error", "cancelrequested", "cancelacknowledged", "cancelled")  # as kernel_state spells them
STOPPED = ("cancelrequested", "cancelacknowledged", "cancelled")  # stopped on Kaggle's page


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--publish", default="", help="copy the run's best network here when it is home")
    ap.add_argument("--hours", type=float, default=11.0, help="pause after this many hours (Kaggle stops at 12)")
    args = ap.parse_args()
    run_dir = Path(args.runs_dir) / args.run
    cloud_file = run_dir / "cloud.json"
    status_file = run_dir / "status.json"
    try:
        status = json.loads(status_file.read_text())
    except (OSError, ValueError):
        status = {}
    config = json.loads((run_dir / "config.json").read_text())

    def set_status(**changes):
        status.update(changes, run=args.run, where="kaggle", pid=os.getpid(), host=socket.gethostname(),
                      generations=status.get("generations") or config.get("generations"), updated=_now())
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        tmp.replace(status_file)

    def save_cloud(**changes):
        data = json.loads(cloud_file.read_text()) if cloud_file.exists() else {}
        data.update(changes)
        cloud_file.write_text(json.dumps(data, indent=2))
        return data

    creds = kaggle_credentials()
    if not creds:
        set_status(state="failed", error="Kaggle is not connected: add your API token in New run.")
        sys.exit(1)
    cloud = json.loads(cloud_file.read_text()) if cloud_file.exists() else {}
    kid = cloud.get("kernel")

    try:
        # A session not yet home (this process was restarted) is followed, or fetched if it has
        # ended, never repeated: a new one starts only when the last one is home.
        state = kernel_state(kid) if kid and not cloud.get("fetched") else None
        if state is None:
            set_status(state="uploading", phase_started=_now(), error=None)
            print(f"Uploading {args.run} to Kaggle...", flush=True)
            kid = push_run(run_dir, creds["username"], args.hours)
            cloud = save_cloud(kernel=kid, url=kernel_url(creds["username"], args.run), pushed=_now(), fetched=False)
            state = "queued"
        if state not in ENDED:
            set_status(state="queued" if state == "queued" else "cloud_running", kernel_url=cloud.get("url"),
                       cloud_started=cloud.get("pushed"), cloud_checked=_now())
        while state not in ENDED:
            time.sleep(CHECK_EVERY)
            try:
                state = kernel_state(kid)
            except KaggleError as exc:  # a passing network problem: keep following
                print(f"Could not check Kaggle: {exc}", flush=True)
                continue
            set_status(state="queued" if state == "queued" else "cloud_running", cloud_checked=_now())
        set_status(state="downloading", cloud_checked=_now())
        print(f"Kaggle session ended ({state}); downloading the run...", flush=True)
        log, came = fetch_output(kid, run_dir)
        save_cloud(fetched=True, ended=_now(), outcome=state)
    except KaggleError as exc:
        set_status(state="failed", error=f"Kaggle: {exc}")
        print(f"Kaggle: {exc}", flush=True)
        sys.exit(1)

    # The run's own status came back with it: keep it, marked as a Kaggle run that is home.
    try:
        status = json.loads(status_file.read_text())
    except (OSError, ValueError):
        status = {}
    if state in STOPPED and came and status.get("state") not in ("finished", "paused"):
        # Stopped on Kaggle's page, its saved work home: stopped like a run closed here, to resume.
        status["state"], status["error"] = "stopped", None
    elif state != "complete" or status.get("state") not in ("finished", "paused"):
        status["state"] = "failed"
        if state in STOPPED:
            status["error"] = ("The session was stopped on Kaggle, and Kaggle kept none of its files, so its "
                               "training is lost. Resume starts again from where the run was before it.")
        else:
            status["error"] = f"The Kaggle session ended with '{state}'" + (f"; see {log.name}." if log else ".")
    best = run_dir / "checkpoints" / "best.pt"
    if args.publish and best.exists():
        Path(args.publish).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, args.publish + ".tmp")
        os.replace(args.publish + ".tmp", args.publish)
    set_status(**{k: v for k, v in status.items() if k not in ("pid", "host", "updated")})
    print(f"{args.run} is home: {status.get('state')}.", flush=True)


if __name__ == "__main__":
    main()
