"""A training run's report: what it learned (the same as the Run report on the Training page).

    python scripts/report.py                      # runs/main: summary, material probe and charts
    python scripts/report.py --run quick          # another run
    python scripts/report.py --strength           # also play the untrained, best and final networks
                                                  # against Random and Greedy (a few minutes)
    python scripts/report.py --strength --games 40

The summary and probe take a second or two. The strength check is saved in
runs/<run>/report/strength.json, and the charts in runs/<run>/training.png.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chaturanga.analysis import (format_probe, format_strength, measure_strength, read_strength,  # noqa: E402
                                 run_summary, value_probe)

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--strength", action="store_true", help="play the strength check (a few minutes)")
    ap.add_argument("--strength-only", action="store_true", help="only the strength check (used by the app)")
    ap.add_argument("--games", type=int, default=20, help="strength check games per network and opponent (default 20)")
    ap.add_argument("--workers", type=int, default=0, help="processes for the strength check (default: cores - 2)")
    args = ap.parse_args()
    run_dir = os.path.join(args.runs_dir, args.run)
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        sys.exit(f"There is no run named {args.run!r} in {args.runs_dir}.")

    if not args.strength_only:
        from chaturanga.analysis import format_summary

        print(f"Run {args.run}\n")
        print(format_summary(run_summary(run_dir)))
        print("\nDoes the value head see material? (value for White to move)\n")
        print(format_probe(value_probe(run_dir)), flush=True)
        print()
        try:
            import subprocess

            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "plot_training.py"),
                            "--run", args.run, "--runs-dir", args.runs_dir], check=True)
        except (subprocess.CalledProcessError, OSError):
            print("(Charts skipped: pip install -e '.[train]' for matplotlib.)")
        saved = read_strength(run_dir)
        if saved and saved.get("state") == "done" and not args.strength:
            print("\nStrength check (saved)\n")
            print(format_strength(saved))
    if args.strength or args.strength_only:
        print("\nStrength check\n", flush=True)
        measure_strength(run_dir, games=args.games, workers=args.workers)


if __name__ == "__main__":
    main()
