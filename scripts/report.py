"""A training run's report: what it learned (the same as Make report on the Training page).

    python scripts/report.py                      # runs/main: summary, material probe and charts
    python scripts/report.py --run quick          # another run
    python scripts/report.py --strength           # also play the untrained, best and final networks
                                                  # against Random and Greedy (a few minutes)
    python scripts/report.py --strength --games 40
    python scripts/report.py --ladder             # also the rating ladder and conversion test
                                                  # (every 10th generation, 40 games a pair)
    python scripts/report.py --ladder --every 5 --games 60 --gpu-workers 10

The report takes a few seconds and is saved in runs/<run>/report/report.json (the Training page
shows it), the charts in runs/<run>/training.png, the strength check in
runs/<run>/report/strength.json and the ladder in runs/<run>/report/benchmark.json. Rating more
generations later plays only their games.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chaturanga.analysis import (format_probe, format_strength, format_summary, make_report,  # noqa: E402
                                 measure_strength, read_strength)
from chaturanga.benchmark import format_benchmark, read_benchmark, run_benchmark  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--strength", action="store_true", help="play the strength check (a few minutes)")
    ap.add_argument("--strength-only", action="store_true", help="only the strength check (used by the app)")
    ap.add_argument("--ladder", action="store_true", help="play the rating ladder and conversion test")
    ap.add_argument("--ladder-only", action="store_true", help="only the ladder and conversion test (used by the app)")
    ap.add_argument("--games", type=int, help="games per network and opponent (strength check, default 20) "
                                              "or per pair (ladder, default 40)")
    ap.add_argument("--every", type=int, default=10, help="rate every n-th generation (default 10)")
    ap.add_argument("--workers", type=int, default=0, help="processes to play with (default: cores - 2)")
    ap.add_argument("--gpu-workers", type=int, default=0, help="of them, how many run the networks on the GPU")
    args = ap.parse_args()
    run_dir = os.path.join(args.runs_dir, args.run)
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        sys.exit(f"There is no run named {args.run!r} in {args.runs_dir}.")

    if not (args.strength_only or args.ladder_only):
        report = make_report(run_dir)  # saved with the run, as the app's Make report does
        print(f"Run {args.run}\n")
        print(format_summary(report["summary"]))
        print("\nDoes the value head see material? (value for White to move)\n")
        print(format_probe(report["probe"]))
        print(f"\nCharts: {os.path.join(run_dir, 'training.png')}" if report["charts"]
              else "\n(Charts skipped: pip install -e '.[train]' for matplotlib.)")
        saved = read_strength(run_dir)
        if saved and saved.get("state") == "done" and not args.strength:
            print("\nStrength check (saved)\n")
            print(format_strength(saved))
        ladder = read_benchmark(run_dir)
        if ladder and ladder.get("state") == "done" and not args.ladder:
            print("\n" + format_benchmark(ladder))
    if (args.strength or args.strength_only) and not args.ladder_only:
        print("\nStrength check\n", flush=True)
        measure_strength(run_dir, games=args.games or 20, workers=args.workers, gpu_workers=args.gpu_workers)
    if args.ladder or args.ladder_only:
        print("\nRating ladder and conversion test\n", flush=True)
        run_benchmark(run_dir, games=args.games or 40, every=args.every, workers=args.workers,
                      gpu_workers=args.gpu_workers)


if __name__ == "__main__":
    main()
