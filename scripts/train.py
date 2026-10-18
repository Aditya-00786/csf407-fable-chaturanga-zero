"""Milestone 3: train the agent by self-play.

    python scripts/train.py                          # run "main": 20 generations of 50 games
    python scripts/train.py --generations 40         # resume "main" and keep going to 40
    python scripts/train.py --run quick --games 8 --generations 3 --steps 50   # a quick test run
    python scripts/train.py --workers 0 --concurrent 64   # one process, batched on the GPU (Colab)

Stop with Ctrl+C at any time; the same command resumes from the last finished generation. The
best network is copied to checkpoints/best.pt, where the web app's ChaturangZero opponent finds it.
Losses and results go to runs/<run>/metrics.jsonl; plot them with scripts/plot_training.py.
"""

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from chaturanga.training import TrainConfig, TrainingRun, training_device  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Command-line flag -> TrainConfig field.
OPTIONS = {
    "generations": "generations", "games": "games_per_generation", "sims": "simulations",
    "steps": "steps_per_generation", "batch": "batch_size", "lr": "learning_rate",
    "buffer": "buffer_positions", "eval_games": "eval_games", "concurrent": "concurrent", "reuse": "sample_reuse",
    "blocks": "blocks", "channels": "channels", "seed": "seed",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main", help="run name (default main)")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"), help="where runs are kept (default runs/)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="self-play processes (default: cores - 2); 0 plays in this process, for a GPU")
    ap.add_argument("--device", help="training device: cuda, mps or cpu (default: the best available)")
    ap.add_argument("--publish", default=os.path.join(ROOT, "checkpoints", "best.pt"),
                    help="copy the best network here for the web app (default checkpoints/best.pt; '' to skip)")
    defaults = TrainConfig()
    for flag, name in OPTIONS.items():
        ap.add_argument(f"--{flag.replace('_', '-')}", dest=flag, type=type(getattr(defaults, name)),
                        help=f"default {getattr(defaults, name)}")
    args = ap.parse_args()

    run_dir = os.path.join(args.runs_dir, args.run)
    config_file = os.path.join(run_dir, "config.json")
    if os.path.exists(config_file):
        import json

        config = TrainConfig.from_json(json.load(open(config_file)))
    else:
        config = defaults
    changes = {name: getattr(args, flag) for flag, name in OPTIONS.items() if getattr(args, flag) is not None}
    config = replace(config, **changes)

    device = torch.device(args.device) if args.device else training_device()
    TrainingRun(run_dir, config, workers=args.workers, device=device, publish=args.publish or None).run()


if __name__ == "__main__":
    main()
