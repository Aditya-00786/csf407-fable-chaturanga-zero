"""Team self-play: generate games on your own machine for someone else's training run.

The person running the training shares a checkpoint (runs/<run>/checkpoints/gen_NNNN.pt, or
best.pt). You play games with it and send back the file this writes; they drop it into
runs/<run>/incoming/, and the run adds the games at its next generation after checking that
every move and visit count is legal.

    python scripts/selfplay.py --checkpoint gen_0005.pt --games 50 --name aadit
    # -> selfplay-aadit-gen0005-<time>.jsonl.gz

Games from the newest network help most; those from a few generations back are still useful.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import multiprocessing as mp  # noqa: E402

from chaturanga.net import load_checkpoint  # noqa: E402
from chaturanga.selfplay import GameRecord, summarise, write_games  # noqa: E402
from chaturanga.training import TrainConfig, _init_worker, selfplay_task  # noqa: E402


def _play(batch: tuple) -> list[dict]:
    return selfplay_task(*batch)


def main():
    defaults = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the network to play with (a .pt file)")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--name", required=True, help="your name, recorded with every game")
    ap.add_argument("--out", help="output file (default: selfplay-<name>-gen<N>-<time>.jsonl.gz)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sims", type=int, default=defaults.simulations, help="must match the run (default %(default)s)")
    ap.add_argument("--seed", type=int, help="default: from the clock, so teammates never repeat each other's games")
    args = ap.parse_args()

    _, meta = load_checkpoint(args.checkpoint)
    generation = int(meta.get("generation", 0))
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 2**31
    out = args.out or f"selfplay-{args.name}-gen{generation:04d}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl.gz"
    config = replace(defaults.selfplay_config, simulations=args.sims)
    print(f"Playing {args.games} games with generation {generation} ({meta.get('run', 'unknown run')}) "
          f"on {args.workers} processes...")

    chunk = max(1, min(config.concurrent, math.ceil(args.games / (2 * args.workers))))
    records: list[GameRecord] = []
    t0 = time.perf_counter()
    batches = [(os.path.abspath(args.checkpoint), list(range(seed + start, seed + min(start + chunk, args.games))),
                config, generation, args.name) for start in range(0, args.games, chunk)]
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
    try:
        for result in pool.imap_unordered(_play, batches):
            records.extend(GameRecord.from_json(d) for d in result)
            print(f"  {len(records)}/{args.games} games ({time.perf_counter() - t0:.0f} s)", flush=True)
        pool.close()
    except KeyboardInterrupt:  # keep what has finished: those games are still worth sending
        print(f"Stopped; keeping the {len(records)} finished games.")
        pool.terminate()
    pool.join()
    if not records:
        sys.exit("No games finished, so nothing was written.")
    records.sort(key=lambda r: r.seed)
    write_games(out, records)
    s = summarise(records)
    print(f"Wrote {out}: {s['games']} games, {s['positions']:,} positions "
          f"(W {s['white_wins']} / D {s['draws']} / B {s['black_wins']}). Send this file to whoever runs the training.")


if __name__ == "__main__":
    main()
