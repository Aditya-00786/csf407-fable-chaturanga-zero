"""Share training runs on GitHub: each run's small results in the repository, and the whole run
(networks and self-play games included) as one archive per run for a GitHub release.

    python scripts/share_runs.py main-v4 main-v3     # writes results/<run>/ and dist/runs/<run>.tar

results/<run>/ holds the run's settings, metrics, log, charts and reports (strength check, rating
ladder): a few MB, enough to redraw its charts and check the numbers in docs/. dist/runs/<run>.tar
holds everything else too; upload the archives to the release `training-runs`:

    gh release upload training-runs dist/runs/*.tar --clobber

and bring a run back into the app with

    gh release download training-runs -p main-v4.tar
    tar -xf main-v4.tar -C runs/

Left out of both: what ties a run to the machines it ran on (launch.json and server.json, with the
server's address and folder, the followers' logs, pause requests), and in the status and reports
the machines' names and process ids. A run brought back this way shows as finished on this Mac.
"""

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files that belong to the machines a run used, never shared.
MACHINE_FILES = {"launch.json", "server.json", "server_run.log", "control.json", "kaggle_run.log", "cloud.json"}
MACHINE_REPORT_FILES = {"server_job.log"}
# In the status and reports: who ran it, where, and how it was followed.
MACHINE_KEYS = {"host", "pid", "where", "server_host", "follower_pid", "follower_host", "remote_pid", "connected",
                "checked"}
# The small results kept in the repository.
RESULTS = ["config.json", "metrics.jsonl", "train.log", "training.png", "report/report.json",
           "report/strength.json", "report/benchmark.json"]


def _skipped(rel: Path) -> bool:
    name = rel.name
    return (name in MACHINE_FILES and len(rel.parts) == 1) or (rel.parts[0] == "report" and name in MACHINE_REPORT_FILES) \
        or name == ".DS_Store" or name.endswith(".tmp") or ".partial." in name


def _clean(rel: Path, data: bytes) -> bytes:
    """The status and report files without the machines' details; anything else as it is."""
    if rel.suffix == ".json" and (rel == Path("status.json") or rel.parts[0] == "report"):
        d = json.loads(data)
        return json.dumps({k: v for k, v in d.items() if k not in MACHINE_KEYS}).encode()
    return data


def export_results(run: Path, out: Path) -> list[str]:
    """Copy the run's small results into `out` (results/<run>/); the files written."""
    if out.exists():
        shutil.rmtree(out)
    written = []
    for name in RESULTS:
        src = run / name
        if src.is_file():
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_clean(Path(name), src.read_bytes()))
            written.append(name)
    return written


def pack(run: Path, out: Path) -> int:
    """Write the whole run, but for what belongs to its machines, as one tar (members <run>/...);
    its size."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w") as tar:
        for path in sorted(run.rglob("*")):
            rel = path.relative_to(run)
            if not path.is_file() or _skipped(rel):
                continue
            arcname = f"{run.name}/{rel.as_posix()}"
            if rel.suffix == ".json" and (rel == Path("status.json") or rel.parts[0] == "report"):
                data = _clean(rel, path.read_bytes())
                info = tarfile.TarInfo(arcname)
                info.size, info.mtime = len(data), int(path.stat().st_mtime)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(path, arcname=arcname, recursive=False)
    return out.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names")
    ap.add_argument("--runs-dir", default=os.environ.get("CHATURANGA_RUNS", str(ROOT / "runs")))
    ap.add_argument("--results", default=str(ROOT / "results"), help="where the small results go (default results/)")
    ap.add_argument("--dist", default=str(ROOT / "dist" / "runs"), help="where the archives go (default dist/runs/)")
    ap.add_argument("--no-archive", action="store_true", help="only the small results")
    args = ap.parse_args()
    for name in args.runs:
        run = Path(args.runs_dir) / name
        if not (run / "config.json").is_file():
            sys.exit(f"There is no run named {name!r} in {args.runs_dir}.")
        files = export_results(run, Path(args.results) / name)
        print(f"{name}: {len(files)} result files in {Path(args.results) / name}")
        if not args.no_archive:
            size = pack(run, Path(args.dist) / f"{name}.tar")
            print(f"{name}: {size / 1e6:,.0f} MB archive at {Path(args.dist) / f'{name}.tar'}")


if __name__ == "__main__":
    main()
